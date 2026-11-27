"""Register one folder of same-camera stills on CPU and freeze its accepted reconstruction."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def register(raw, output, threads):
    import pycolmap

    from recovery.data import freeze

    raw, output = raw.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(p.name for p in raw.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file())
    if not images or threads < 1:
        raise ValueError("Provide a folder of original images and a positive thread count.")
    from recovery.data import _sha
    identity = dict(files={n:_sha(raw/n) for n in images}, pycolmap=pycolmap.__version__,
                    camera="shared SIMPLE_RADIAL", matching="exhaustive SIFT", split_seed=42)
    inventory = output/"registration-input.json"
    if inventory.exists():
        if json.loads(inventory.read_bytes()) != identity:
            raise ValueError("The registration input changed. Use a fresh output folder.")
    else:
        if any(output.iterdir()):
            raise ValueError("Use an empty output folder for a new registration.")
        inventory.write_text(json.dumps(identity, indent=2), encoding="utf8")
    db = output/"features.db"
    if not (output/"matching-complete").exists():
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = threads
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        pycolmap.extract_features(db, raw, image_names=images, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = threads
        pycolmap.match_exhaustive(db, matching_options=matching, device=pycolmap.Device.cpu)
        (output/"matching-complete").touch()
    models = output/"sparse"
    models.mkdir(exist_ok=True)
    if not (output/"mapping-complete.json").exists():
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = threads
        options.random_seed = 42
        found = pycolmap.incremental_mapping(db, raw, models, options=options)
        if not found:
            raise ValueError("COLMAP did not reconstruct a connected camera model.")
        (output/"mapping-complete.json").write_text(json.dumps(sorted(found)), encoding="utf8")
    reconstructions = {i:pycolmap.Reconstruction(models/str(i)) for i in json.loads((output/"mapping-complete.json").read_bytes())}
    idx, rec = max(reconstructions.items(), key=lambda item:(item[1].num_reg_images(),-item[1].compute_mean_reprojection_error()))
    registered = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im:im.name)
    errors = [p.error for p in rec.points3D.values()]
    summary = dict(raw=len(images), registered=len(registered), model=idx, model_sizes={i:r.num_reg_images() for i,r in reconstructions.items()},
                   points=rec.num_points3D(), mean_point_error_px=float(np.mean(errors)),
                   missing=sorted(set(images)-{im.name for im in registered}))
    (output/"registration.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(summary),flush=True)
    if len(registered)/len(images)<.95 or not 0<=summary["mean_point_error_px"]<1:
        raise ValueError("Capture did not pass the 95% registration and subpixel mean point-error gates. See registration.json.")
    if len(rec.cameras)!=1:
        raise ValueError("The renderer requires shared camera intrinsics.")
    camera = next(iter(rec.cameras.values()))
    if camera.model_name!="SIMPLE_RADIAL":
        raise ValueError("Expected a shared SIMPLE_RADIAL camera.")
    focal,cx,cy,k1 = map(float,camera.params)
    scene=output/"scene"
    (scene/"images").mkdir(parents=True,exist_ok=True)
    frames=[]
    # COLMAP world coordinates are retained. Flip camera Y and Z to convert
    # camera-to-world axes from OpenCV (right/down/forward) to OpenGL.
    conversion=np.diag([1.,-1.,-1.,1.])
    for im in registered:
        c2w=np.eye(4)
        c2w[:3]=im.cam_from_world().inverse().matrix()
        name=Path("images")/im.name
        shutil.copyfile(raw/im.name,scene/name)
        frames.append(dict(file_path=name.as_posix(),transform_matrix=(c2w@conversion).tolist(),colmap_im_id=im.image_id))
    meta=dict(w=camera.width,h=camera.height,fl_x=focal,fl_y=focal,cx=cx,cy=cy,k1=k1,k2=0.,p1=0.,p2=0.,
              camera_model="OPENCV",applied_transform=np.eye(4)[:3].tolist(),frames=frames)
    (scene/"transforms.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    text_model=output/"colmap-text"
    text_model.mkdir(exist_ok=True)
    rec.write_text(text_model)
    m=freeze("personal",scene,output/"personal.json",raw=raw,points=text_model/"points3D.txt",test_count=12,split_seed=42)
    print("Frozen",m["data_hash"],{k:len(v) for k,v in m["splits"].items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    register(args.raw,args.output,args.threads)

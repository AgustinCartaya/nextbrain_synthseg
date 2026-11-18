import os
import subprocess

def run_fastsurfer_docker(t1_path, subj_id, out_dir, gpu=0, license_path=None, verify=False, verbose=False):
    """
    Ejecuta FastSurfer dentro de un contenedor Docker sin crear carpetas root.

    Args:
        t1_path (str): Ruta absoluta a la imagen T1w (NIfTI o mgz).
        subj_id (str): ID del sujeto (nombre usado en SUBJECTS_DIR).
        out_dir (str): Carpeta donde se guardarán los resultados (subjects_dir).
        gpu (int): ID de la GPU a usar.
        license_path (str): Ruta a tu license.txt de FreeSurfer (requerido).
        verify (bool): Si True, evita volver a correr si ya existe salida.
        verbose (bool): Si True, imprime logs del contenedor.
    """
    subj_out_path = os.path.join(out_dir, subj_id)
    if verify and os.path.exists(subj_out_path):
        print(f"FastSurfer ya corrido para: {subj_id}\nResultados en: {subj_out_path}")
        return True
    
    if license_path is None:
        raise ValueError("Debes especificar la ruta al license.txt de FreeSurfer")

    # Obtener UID y GID actuales
    uid = os.getuid()
    gid = os.getgid()

    # Construimos el comando docker (igual al que comprobaste que funciona)
    command = [
        "docker", "run", "--rm",
        "--gpus", f"device={gpu}",
        "--user", f"{uid}:{gid}",               # evita permisos root
        "-v", f"{os.path.abspath(t1_path)}:/input/input.nii.gz:ro",
        "-v", f"{os.path.abspath(out_dir)}:/output",
        "-v", f"{os.path.abspath(license_path)}:/license.txt:ro",
        "deepmi/fastsurfer:latest",
        "--t1", "/input/input.nii.gz",
        "--sid", subj_id,
        "--sd", "/output",
        "--parallel",
        "--fs_license", "/license.txt",
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            print(f"FastSurfer finalizado para {subj_id}. Resultados en: {subj_out_path}")
            print(result.stdout)
        return True
    else:
        print("Error ejecutando FastSurfer")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        return False


### ---- Example to run with python
# run_fastsurfer_docker(
#     t1_path="/home/agustin/phd/synthesis/preprocessing/test_preprocessing/results/fastsurfer/input/sub-ADNI002S0295_ses-m000_run-01_T1w.nii.gz",
#     subj_id="ADNI002S0295_python",
#     out_dir="/home/agustin/phd/synthesis/preprocessing/test_preprocessing/results/fastsurfer/output",
#     gpu=2,
#     license_path="/home/agustin/phd/synthesis/preprocessing/freesurfer_license/license.txt",
#     verify=True,
#     verbose=True
# )

### ---- Example to run in terminal
# docker run --rm \
#     --gpus device=1 \
#     --user $(id -u):$(id -g) \
#     -v /home/agustin/phd/synthesis/preprocessing/test_preprocessing/results/fastsurfer/input/sub-ADNI002S0295_ses-m000_run-01_T1w.nii.gz:/input/input.nii.gz:ro \
#     -v /home/agustin/phd/synthesis/preprocessing/test_preprocessing/results/fastsurfer/output:/output \
#     -v /home/agustin/phd/synthesis/preprocessing/freesurfer_license/license.txt:/license.txt:ro \
#     deepmi/fastsurfer:latest \
#     --t1 /input/input.nii.gz \
#     --sid ADNI002S0295 \
#     --sd /output \
#     --parallel \
#     --fs_license /license.txt



run_fastsurfer_docker("/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/correct_cortex/synthetic_images_creation/generated/synthetic_t1w/images/case-1.nii.gz",
                     "case-1", 
                     "/home/agustin/phd/synthseg_nextbrain/project/dev/wp2/correct_cortex/synthetic_images_creation/generated/synthetic_t1w/fastsurfer", 
                     gpu=0, 
                     license_path="/home/agustin/licenses/fastsurfer.txt", 
                     verify=False, 
                     verbose=True)

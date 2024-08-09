# If the CT and masks are already available as NIFTIs
# it's only necessary to perform the FSL operations.
# This was the case with the dataset from Maastro.
import csv
import os

from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

FSL_DIR = "FSLDIR"

if __name__ == '__main__':
    # Temporary folder to save the NIFTIs
    temporary_folder = './niftis/'
    # Input for the docker container running FSL
    input_folder = '/mnt/input'
    output_folder = '/mnt/output'
    bash_command = []
    # Example of the input file in '/example/nifti_files_maastro.csv'
    input_file = ''
    with open(input_file, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            id, mask_path, scan_path, exclude = row
            print(id)
            if exclude:
                print("Excluded")
            else:
                gtv_masks = onlyfiles = [
                    mask for mask in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f)) and  "GTV" in f.upper()
                ]
                if len(gtv_masks) == 0:
                    print(f"GTV mask not found for patient: {id}")
                else:
                    # Re-orient the scan
                    bash_command.append(
                        ' '.join(
                            os.getenv(FSL_DIR) + "/bin/fslreorient2std",
                            scan_path,
                            temporary_folder + id + "/image_re.nii.gz",
                        )
                    )
                    # Re-orient the mask
                    masks_reorient = []
                    for mask in gtv_masks:
                        masks_reorient.append(
                            ' '.join(
                                os.getenv(FSL_DIR) + "/bin/fslreorient2std",
                                mask_path + mask,
                                # temporary_folder + id + f"/mask_{mask}_re.nii.gz",
                                temporary_folder + id + f"/mask_{mask}_re.nii.gz",
                            )
                        )
                    # Add multiple masks
                    if len(gtv_masks) > 1:
                        bash_command.append(
                            ' '.join(
                                os.getenv(FSL_DIR) + "/bin/fslmaths",
                                " -add".join([temporary_folder + id + f"/mask_{mask}_re.nii.gz" for mask in gtv_masks]),
                                temporary_folder + id + f"/mask_{''.join(gtv_masks)}_re.nii.gz",
                            )
                        )  
                    # Subtract the mask
                    bash_command.append(
                        ' '.join(
                            os.getenv(FSL_DIR) + "/bin/fslmaths",
                            temporary_folder + id + "/image_re.nii.gz",
                            "-mas",
                            temporary_folder + id + f"/mask_{''.join(gtv_masks)}_re.nii.gz",
                            output_folder + f"/{id}.nii.gz",
                        )
                    )

    with open('fsl_script.sh', 'w') as f:
        for line in bash_command:
            f.write(f"{line}\n")

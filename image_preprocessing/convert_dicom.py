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
    # Example of the input file in '/example/dicom_files.csv'
    input_file = ''
    with open(input_file, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            id, mask_path, scan_path, exclude = row
            print(id)
            if exclude:
                print("Excluded")
            else:
                # Convert the DICOM samples to NIFTIs
                structures = list_rt_structs(mask_path)
                gtv_masks = [structure for structure in structures if "GTV" in structure.upper()]
                if len(gtv_masks) == 0:
                    print("GTV mask not found")
                    print(structures)
                else:
                    dcmrtstruct2nii(
                        mask_path,
                        scan_path,
                        temporary_folder + id,
                        convert_original_dicom=True,
                        structures=gtv_masks,
                    )
                    # Re-orient the scan
                    bash_command.append(
                        ''.join(
                            os.getenv(FSL_DIR) + "/bin/fslreorient2std",
                            temporary_folder + id + "/image.nii.gz",
                            temporary_folder + id + "/image_re.nii.gz",
                        )
                    )
                    # Re-orient the mask
                    for mask in structures:
                        bash_command.append(
                            ''.join(
                                os.getenv(FSL_DIR) + "/bin/fslreorient2std",
                                temporary_folder + id + f"/mask_{mask}.nii.gz",
                                temporary_folder + id + f"/mask_{mask}_re.nii.gz",
                            )
                        )
                    # Add multiple masks
                    if len(structures) > 1:
                        bash_command.append(
                            ''.join(
                                os.getenv(FSL_DIR) + "/bin/fslmaths",
                                " -add".join([temporary_folder + id + f"/mask_{mask}_re.nii.gz" for mask in structures]),
                                temporary_folder + id + f"/mask_{''.join(structures)}_re.nii.gz",
                            )
                        )  
                    # Subtract the mask
                    bash_command.append(
                        ''.join(
                            os.getenv(FSL_DIR) + "/bin/fslmaths",
                            temporary_folder + id + "/image_re.nii.gz",
                            "-mas",
                            temporary_folder + id + f"/mask_{''.join(structures)}_re.nii.gz",
                            output_folder + f"/{id}.nii.gz",
                        )
                    )

    with open('fsl_script.sh', 'w') as f:
        for line in bash_command:
            f.write(f"{line}\n")

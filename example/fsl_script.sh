/usr/share/fsl/5.0/bin/fslreorient2std /mnt/scans/HN-CHUM-001/image.nii.gz /mnt/output2/HN-CHUM-001_im.nii.gz
/usr/share/fsl/5.0/bin/fslreorient2std /mnt/scans/HN-CHUM-001/mask_GTV.nii.gz /mnt/output2/HN-CHUM-001_mask.nii.gz
/usr/share/fsl/5.0/bin/fslmaths /mnt/output2/HN-CHUM-001_im.nii.gz -mas /mnt/outputmasks/HN-CHUM-001_mask.nii.gz /mnt/outputscans/HN-CHUM-001.nii.gz
/usr/share/fsl/5.0/bin/fslreorient2std /mnt/scans/HN-CHUM-002/image.nii.gz /mnt/output2/HN-CHUM-002_im.nii.gz
/usr/share/fsl/5.0/bin/fslreorient2std /mnt/scans/HN-CHUM-002/mask_GTV-et-GG.nii.gz /mnt/output2/HN-CHUM-002_mask.nii.gz
/usr/share/fsl/5.0/bin/fslmaths /mnt/output2/HN-CHUM-002_im.nii.gz -mas /mnt/outputmasks/HN-CHUM-002_mask.nii.gz /mnt/outputscans/HN-CHUM-002.nii.gz

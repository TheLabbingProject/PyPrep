from pathlib import Path
from PyPrep.utils import check_files_existence, FSLOUTTYPE
from logs import messages
from PyPrep.diffusion import dmri_prep_functions as dmri_methods


class DenoiseDWI:
    def __init__(self, epi_file: Path, mask: Path):
        """
        Perform mrtrix's initial denoising.
        Arguments:
            epi_file {Path} -- [Path to dwi file]
            mask {Path} -- [Path to dwi brain mask]
        """
        self.epi_file = epi_file
        self.mask = mask
        self.out_file = Path(epi_file.parent / f"{epi_file.stem}_denoised.mif")
        self.exist = check_files_existence([self.out_file])

    def __str__(self):
        str_to_print = messages.DENOISEDWI.format(
            epi_file=self.epi_file.name,
            mask=self.mask.name,
            denoised=self.out_file.name,
        )

    def denoise(self):
        denoiser = dmri_methods.denoise_dwi(self.epi_file, self.mask, self.out_file)
        denoiser.run()

    def run(self):
        if not self.exist:
            print("Performing initial denoising procedure...")
            self.denoise()
        else:
            print("Initial denoised procedure already done. Continuing.")
        return self.out_file

from pathlib import Path
from PyPrep.utils import check_files_existence, FSLOUTTYPE
from logs import messages
from PyPrep.diffusion import dmri_prep_functions as dmri_methods


class Unring:
    def __init__(self, denoised: Path):
        """
        Use Mrtrix3's tools for Gibbs rings removal
        Arguments:
            denoised {Path} -- [Initally denoised DWI image]
        """
        self.denoised = denoised
        self.out_file = Path(denoised.with_name(denoised.stem + "_degibbs.mif"))
        self.exist = check_files_existence([self.out_file])

    def __str__(self):
        str_to_print = messages.UNRING.format(
            denoised=self.denoised.name, degibbs=self.out_file.name
        )

    def unring(self):
        unring = dmri_methods.unring_dwi(self.denoised, self.out_file)
        unring.run()

    def run(self):
        if not self.exist:
            print("Performing Gibbs rings removal for DWI data")
            self.unring()
        else:
            print("Gibbs rings removal already done. Continuing.")
        return self.out_file

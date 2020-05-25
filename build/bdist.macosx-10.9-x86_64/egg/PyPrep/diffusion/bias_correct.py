from pathlib import Path
from PyPrep.utils import check_files_existence, FSLOUTTYPE
from logs import messages
from PyPrep.diffusion import dmri_prep_functions as dmri_methods


class BiasCorrect:
    def __init__(self, preprocessed: Path):
        self.preprocessed = preprocessed
        self.out_file = Path(preprocessed.parent / f"{preprocessed.stem}_biascorr.mif")
        self.exist = check_files_existence([self.out_file])

    def __str__(self):
        str_to_print = messages.BIASCORRECTION.format(
            preprocessed=self.preprocessed, bias_corrected=self.out_file
        )
        return str_to_print

    def bias_correct(self):
        self.out_file = dmri_methods.bias_correct(self.preprocessed, self.out_file)

    def run(self):
        if not self.exist:
            print("Performing initial B1 bias field correction of DWIs")
            self.bias_correct()
        else:
            print("Initial B1 bias field correction already done. Continuing.")
        return self.out_file

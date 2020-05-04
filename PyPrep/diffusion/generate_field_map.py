from pathlib import Path
from PyPrep.utils import check_files_existence, FSLOUTTYPE
from logs import messages
from PyPrep.functional import fmri_prep_functions as fmri_methods
from PyPrep.utils.brain_extraction import BrainExtraction


class GenerateFieldMap:
    """
    Generate Field Maps using dwi's AP and phasediff's PA scans
    Arguments:
        AP {Path} -- [path to dwi's AP file]
        PA {Path} -- [path to PA phasediff file]
        outdir {Path} -- [path to output directory (outdir/sub-xx/fmap)]
    """

    def __init__(self, subj: str, AP: Path, PA: Path, derivatives_dir: Path):
        self.subj = subj
        self.AP = AP
        self.PA = PA
        self.out_dir = derivatives_dir / subj / "fmap"
        self.initiate_output_files()

    def initiate_output_files(self):
        self.merged = self.out_dir / f"merged_phasediff{FSLOUTTYPE}"
        self.datain = self.out_dir / "datain.txt"
        self.index_file = self.out_dir / "index.txt"
        self.fieldmap = self.out_dir / f"fieldmap{FSLOUTTYPE}"
        self.fieldmap_rad = Path(
            self.fieldmap.parent / f"{Path(self.fieldmap.stem).stem}_rad{FSLOUTTYPE}"
        )
        self.fieldmap_mag = Path(
            self.fieldmap.parent
            / f"{Path(self.fieldmap.stem).stem}_magnitude{FSLOUTTYPE}"
        )
        self.fieldmap_mag_brain = Path(
            self.fieldmap.parent
            / f"{Path(self.fieldmap.stem).stem}_magnitude_brain{FSLOUTTYPE}"
        )

    def __str__(self):
        str_to_print = messages.GENERATEFIELDMAP.format(
            output_dir=self.out_dir,
            AP=self.AP.name,
            PA=self.PA.name,
            datain=self.datain.name,
            index=self.index_file.name,
            fieldmap=self.fieldmap.name,
            fieldmap_rad=self.fieldmap_rad.name,
            fieldmap_mag=self.fieldmap_mag.name,
            fieldmap_brain=self.fieldmap_mag_brain.name,
        )
        return str_to_print

    def merge_phasediff(self):
        """Combine two images into one 4D file
        """
        exist = check_files_existence([self.merged])
        if not exist:
            print("generating dual-phase encoded image.")
            merger = fmri_methods.merge_phases(self.AP, self.PA, self.merged)
            merger.run()
        else:
            print("Dual-phase encoded image already exists. Continuing.")

    def generate_datain(self):
        """Generate datain.txt file for topup
        """
        exist = check_files_existence([self.datain])
        if not exist:
            print("Generating datain.txt with dual-phase data.")
            fmri_methods.generate_datain(self.AP, self.PA, self.datain)
        else:
            print("datain.txt already exists. Continuing.")

    def generate_index(self):
        """Generates index.txt file, needed to run eddy-currents corrections.
        """
        exist = check_files_existence([self.index_file])
        if not exist:
            print("Generating index.txt file, for later eddy-currents correction.")
            fmri_methods.generate_index(self.AP, self.index_file)
        else:
            print("index.txt file already exists. Continuing.")

    def perform_top_up(self):
        """Generate Fieldmap
        """
        exist = check_files_existence([self.fieldmap_mag])
        if not exist:
            print("Using FSL's TopUp to generate fieldmap images.")
            fmri_methods.top_up(self.merged, self.datain, self.fieldmap)
        else:
            print("Fieldmap images already exists. Continuing.")

    def brain_extract(self):
        """Extract fieldmap_mag's brain"""
        exist = check_files_existence([self.fieldmap_mag_brain])
        if not exist:
            print(
                "Using FSL's BET to generate brain-extracted fieldmap magnitude image."
            )
            bet = BrainExtraction(
                in_file=self.fieldmap_mag, out_file=self.fieldmap_mag_brain
            )
            fieldmap_brain = bet.run()

    def run(self):
        for procedure in [
            self.merge_phasediff,
            self.generate_datain,
            self.generate_index,
            self.perform_top_up,
            self.brain_extract,
        ]:
            procedure()
        return self.fieldmap_rad, self.fieldmap_mag_brain, self.index_file, self.datain

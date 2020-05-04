from pathlib import Path
import subprocess
from logs import messages


class CheckBids:
    """
    Validates {bids_dir} as a BIDS compatible directory or raises an error otherwise.

    Keyword Arguments:
        bids_dir {[Path]} -- [Path to a BIDS compliant directory.] (default: {self.mother_dir})
    """

    def __init__(self, bids_dir: Path):
        self.bids_dir = bids_dir

    def __str__(self):
        str_to_print = messages.BIDSVALIDATOR.format(bids_dir=self.bids_dir)
        return str_to_print

    def validate_bids(self):
        try:
            validator = subprocess.check_output(
                ["bids-validator", "--ignoreWarnings", f"{self.bids_dir}"]
            )
        except:
            validator = "Incompatible BIDS directory"
        return validator

    def run(self):
        print(f"Validating {self.bids_dir} as a BIDS compatible directory...")
        validator = self.validate_bids()
        if not "BIDS compatible" in str(validator):
            raise ValueError(
                f"This path is not a BIDS comaptible direcory.\n\t Please make sure it follows the BIDS specifications before going through preprocessing procedures."
            )
        else:
            print("This is a BIDS compliant directory! Moving on.")

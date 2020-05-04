from pathlib import Path
from PyPrep.utils.utillities import (
    check_files_existence,
    FSLOUTTYPE,
    bet_brain_extract,
)
from logs import messages


class BrainExtraction:
    """
    Perform brain extraction using FSL's BET.
    Arguments:
        in_file {Path} -- [Path to input nifti image]
    Keyword Arguments:
        out_file {Path} -- [Path to output nifti image] (default: {None})

    """

    def __init__(self, in_file: Path, out_file: Path = None):
        self.in_file = in_file
        self.out_file = out_file
        if not out_file:
            self.exist = False
        else:
            self.exist = check_files_existence([out_file])

    def __str__(self):
        str_to_print = messages.BETBRAINEXTRACTION.format(
            in_file=self.in_file, out_file=self.out_file
        )

    def brain_extract(self):
        bet, self.out_file = bet_brain_extract(self.in_file, self.out_file)
        bet.run()

    def run(self):
        if not self.exist:
            print("Performing brain extractoin using FSL`s bet")
            self.brain_extract()
        else:
            print("Brain extraction already done.")
            return self.out_file

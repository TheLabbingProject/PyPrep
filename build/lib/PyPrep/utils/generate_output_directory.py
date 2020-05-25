from pathlib import Path
import glob
import os
from logs import messages
from .displayable_path import DisplayablePath


class GenerateOutputDirectory:
    """Generates a derivative'output directory at {derivatives_dir} that resesmbles {bids_dir} tree."""

    def __init__(self, bids_dir: Path, derivatives_dir: Path, subj: str):
        self.bids_dir = bids_dir
        self.derivatives_dir = derivatives_dir
        self.subj = subj

    def generate_directories(self):
        in_name = self.bids_dir.name
        out_name = self.derivatives_dir.name
        dirs = glob.glob(f"{Path(self.bids_dir / self.subj)}/*")
        dirs.append(f"{os.path.dirname(dirs[0])}/atlases")
        dirs.append(f"{os.path.dirname(dirs[0])}/scripts")
        for d in dirs:
            new_d = d.replace(in_name, out_name)
            if not os.path.isdir(new_d):
                os.makedirs(new_d)

    def __str__(self):
        str_to_print = messages.GENERATEOUTPUT.format(
            bids_dir=self.bids_dir, derivatives_dir=self.derivatives_dir
        )
        return str_to_print

    def run(self):
        self.generate_directories()
        print(f"Output directory`s tree:")
        paths = DisplayablePath.make_tree(self.derivatives_dir / self.subj)
        for path in paths:
            print(path.displayable())

import tqdm
import pandas as pd
import os
from processes_mh import BaseProcess
from dn3_data_dataset_mh import Dataset, Thinker


class ThinkerwiseResultTracker:

    def __init__(self):
        """
        Track the performance of :any:`Thinker`(s) under a certain process.
        """
        self._sheets = dict()

    def _update_sheet(self, ds_name, summary):
        if ds_name not in self._sheets:
            self._sheets[ds_name] = list()
        self._sheets[ds_name].append(summary)

    def add_results_thinker(self, process: BaseProcess, ds_name: str, thinker: Thinker, **kwargs):
        metrics = process.evaluate(thinker)
        summary = {'Person': str(thinker.person_id),
                   'Dataset': ds_name,
                   **metrics,
                   **kwargs}
        self._update_sheet(ds_name, summary)

    def add_results_all_thinkers(self, process: BaseProcess, ds_name: str, fold_dataset: Dataset, **kwargs):
        for _, _, test_thinker in tqdm.tqdm(fold_dataset.loso(), total=len(fold_dataset.thinkers)):
            self.add_results_thinker(process, ds_name, test_thinker, **kwargs)

    def performance_summary(self, ds_name):
        if ds_name not in self._sheets:
            print("Could not find {} to create performance summary.".format(ds_name))
        tqdm.tqdm.write(str(pd.DataFrame(self._sheets[ds_name]).describe()))

    def to_spreadsheet(self, filename: str, path = './results'):
        if not os.path.exists(path):
            os.makedirs(path)
        with pd.ExcelWriter(filename) as writer:
            print("Opened", filename)
            for ds_name in self._sheets:
                df = pd.DataFrame(self._sheets[ds_name])
                df.to_excel(writer, sheet_name=ds_name, header=True, index=False)
                print("Wrote results for {}...".format(ds_name))

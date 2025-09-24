import csv, os, time, pathlib, datetime

class RunLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.p2_inference_csv = self.log_dir / "p2_inference_log.csv"
        self.num_iters_csv = self.log_dir / "p2_num_iters_log.csv"

        self._init_csv(self.p2_inference_csv, ["ts", "image", "crop", "iter", "action",
                                    "issues_before", "issues_after", "reward_p2", "initial_objects"])
        self._init_csv(self.num_iters_csv, ["ts", "image", "crop", "initial_objects", "num_iters"])

    def _init_csv(self, path, header):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def log_num_iters(self, image, crop, initial_objects, num_iters):
        with open(self.num_iters_csv, "a", newline="") as f:
            csv.writer(f).writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image, crop, initial_objects, num_iters])

    def log_p2_inference(self, image, crop, iter_idx, action, before, after, reward, initial_objects=0):
        with open(self.p2_inference_csv, "a", newline="") as f:
            csv.writer(f).writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image, crop, iter_idx,
                                    action, before, after, reward, initial_objects])
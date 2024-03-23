# Copyright (c) UD_lab. All rights reserved.
import copy
from collections import OrderedDict

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from sklearn.metrics import average_precision_score


@METRICS.register_module()
class AnimalKingdomMetric(BaseMetric):
    """Custom mAP evaluation metric."""

    default_prefix = "mAP"

    def __init__(self, collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.results = []
        self.segments_info = []

    def process(self, data_batch, data_samples):
        """Process one batch of data samples and data_samples. The processed
        results should be stored in `self.results`, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (dict): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample["pred_score"]
            label = data_sample["gt_label"]
            result["pred"] = pred.cpu().numpy()
            result["label"] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # Separate results by segments
        index_category = dict(
            head=[
                1,
                2,
                15,
                38,
                40,
                48,
                52,
                67,
                68,
                69,
                78,
                90,
                100,
                102,
                104,
                123,
                133,
            ],
            middle=[
                5,
                7,
                8,
                13,
                16,
                25,
                26,
                27,
                32,
                39,
                45,
                47,
                50,
                51,
                58,
                65,
                76,
                80,
                96,
                97,
                103,
                105,
                112,
                114,
                116,
                120,
                121,
                128,
                135,
            ],
            tail=[
                0,
                3,
                4,
                6,
                9,
                10,
                11,
                12,
                14,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                28,
                29,
                30,
                31,
                33,
                34,
                35,
                36,
                37,
                41,
                42,
                43,
                44,
                46,
                49,
                53,
                54,
                55,
                56,
                57,
                59,
                60,
                61,
                62,
                63,
                64,
                66,
                70,
                71,
                72,
                73,
                74,
                75,
                77,
                79,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                91,
                92,
                93,
                94,
                95,
                98,
                99,
                101,
                106,
                107,
                108,
                109,
                110,
                111,
                113,
                115,
                117,
                118,
                119,
                122,
                124,
                125,
                126,
                127,
                129,
                130,
                131,
                132,
                134,
                136,
                137,
                138,
                139,
            ],
        )

        eval_results = OrderedDict()

        # Compute mAP for each segment
        preds = np.array([x["pred"] for x in results])
        labels = np.array([x["label"] for x in results])
        for key, category in index_category.items():
            preds_category = preds[:, category]
            labels_category = labels[:, category]

            preds_category = preds_category[:, ~(np.all(labels_category == 0, axis=0))]
            labels_category = labels_category[
                :, ~(np.all(labels_category == 0, axis=0))
            ]

            aps = [0]
            try:
                aps = average_precision_score(
                    labels_category, preds_category, average=None
                )
            except ValueError:
                print(
                    "Average precision requires a sufficient number of samples \
                    in a batch which are missing in this sample."
                )

            eval_results[f"{key}_mAP"] = np.mean(aps)

        preds = preds[:, ~(np.all(labels == 0, axis=0))]
        labels = labels[:, ~(np.all(labels == 0, axis=0))]
        aps = [0]
        try:
            aps = average_precision_score(labels, preds, average=None)
        except ValueError:
            print(
                "Average precision requires a sufficient number of samples \
                in a batch which are missing in this sample."
            )

        overall_mAP = np.mean(aps)
        eval_results["overall_mAP"] = overall_mAP

        return eval_results

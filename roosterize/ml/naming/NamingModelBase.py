from typing import *

import abc
import collections
import nltk
import numpy as np
from pathlib import Path
import random

from seutil import LoggingUtils, IOUtils
from seutil.MiscUtils import classproperty

from roosterize.FilesManager import FilesManager
from roosterize.data.Lemma import Lemma
from roosterize.Macros import Macros
from roosterize.ml.MLModelBase import MLModelBase
from roosterize.Utils import Utils


TConfig = TypeVar("TConfig")


class NamingModelBase(MLModelBase[TConfig], abc.ABC):

    logger = LoggingUtils.get_logger(__name__)

    @abc.abstractmethod
    def eval_impl(
            self,
            processed_data_dir: Path,
            beam_search_size: int,
            k: int,
    ) -> List[List[Tuple[str, float]]]:
        pass

    BEAM_SEARCH_SIZE = 5
    K = 5

    ALL = "top-1"
    TOP_KS = [2,3,5]

    @classmethod
    def TOP_K(cls, k):  return f"top-{k}"

    @classproperty
    def MNAMES_METRICS(cls) -> Dict[str, Any]:
        return dict(
            **{i: i for i in [
                cls.ALL,
            ] + [cls.TOP_K(k) for k in cls.TOP_KS]},
        )

    @classproperty
    def QUICK_METRICS(cls) -> List[str]:
        ms = [
            cls.ALL,
        ]
        ms.extend([cls.TOP_K(k) for k in cls.TOP_KS])
        return ms

    @classmethod
    def bleu(cls, truths: List[List[str]], prediction: List[str]) -> float:
        return nltk.translate.bleu_score.sentence_bleu(truths, prediction, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)

    @classmethod
    def acc(cls, truth: List[str], prediction: List[str]) -> float:
        return len([i for i in range(min(len(truth), len(prediction))) if truth[i] == prediction[i]])/max(len(truth), len(prediction))

    @classmethod
    def acc_unordered(cls, truth: List[str], prediction: List[str]) -> float:
        return sum((collections.Counter(truth) & collections.Counter(prediction)).values())/max(len(truth), len(prediction))

    def process_data_impl(self,
            data_dir: Path,
            output_processed_data_dir: Path,
    ) -> NoReturn:
        """
        Extracts lemma names to processed data dir (for eval).
        """
        lemmas: List[Lemma] = IOUtils.dejsonfy(IOUtils.load(data_dir/"lemmas.json", IOUtils.Format.json), List[Lemma])

        lemma_names = [l.name for l in lemmas]
        IOUtils.dump(output_processed_data_dir/"lemma-names.json", lemma_names, IOUtils.Format.jsonPretty)
        lemma_qnames = [l.qname for l in lemmas]
        IOUtils.dump(output_processed_data_dir/"lemma-qnames.json", lemma_qnames, IOUtils.Format.jsonPretty)
        data_indexes = [l.data_index for l in lemmas]
        IOUtils.dump(output_processed_data_dir/"data-indexes.json", data_indexes, IOUtils.Format.jsonPretty)
        return

    def eval(
            self,
            processed_data_dir: Path,
            output_result_dir: Path,
    ) -> NoReturn:
        IOUtils.mk_dir(output_result_dir)

        lemma_names: List[str] = IOUtils.load(processed_data_dir/"lemma-names.json", IOUtils.Format.json)
        lemma_qnames: List[str] = IOUtils.load(processed_data_dir/"lemma-qnames.json", IOUtils.Format.json)
        data_indexes: List[str] = IOUtils.load(processed_data_dir/"data-indexes.json", IOUtils.Format.json)
        predictions_results: List[List[Tuple[str, float]]] = self.eval_impl(processed_data_dir, self.BEAM_SEARCH_SIZE, self.K)
        predictions: List[str] = list()

        # Save predictions (for error-analyze)
        IOUtils.dump(output_result_dir/"predictions-results.json", predictions_results, IOUtils.Format.json)

        # Calculate full match accuracy
        counters: Counter[Any] = collections.Counter()
        counters_correct: Counter[Any] = collections.Counter()

        # Char and Fragment accuracy
        char_accuracies: List[float] = list()
        frag_accuracies: List[float] = list()

        # BLEU scores (BLEU-4)
        bleu_scores: List[float] = list()

        for pred_i in range(len(lemma_names)):
            truth = lemma_names[pred_i]
            candidates_logprobs: List[Tuple[str, float]] = predictions_results[pred_i]

            if len(candidates_logprobs) == 0:
                prediction = ""
            else:
                prediction = candidates_logprobs[0][0]
            # end if

            predictions.append(prediction)

            # Increment counter
            counters[self.ALL] += 1
            # top-k accuracy
            for k in self.TOP_KS:  counters[self.TOP_K(k)] += 1

            if prediction == truth:
                # Increment correct counter
                counters_correct[self.ALL] += 1
            # end if

            for k in self.TOP_KS:
                if truth in [x[0] for x in candidates_logprobs[:k]]:
                    # top-k accuracy
                    counters_correct[self.TOP_K(k)] += 1
            # end if

            # Char and Frag acc
            char_accuracies.append(self.acc([c for c in truth], [c for c in prediction]))
            frag_accuracies.append(self.acc_unordered([f for f in truth.split("_")], [f for f in prediction.split("_")]))

            # BLEU score
            bleu_scores.append(self.bleu([[c for c in truth]], [c for c in prediction]))
        # end for

        test_metrics = dict()
        accuracies: Dict[Any, float] = dict()
        for mname, m in self.MNAMES_METRICS.items():
            if counters[m] != 0:
                accuracies[m] = counters_correct[m] / counters[m]
            else:
                accuracies[m] = np.NaN
            # end if

            test_metrics[f"count-{mname}"] = counters[m]
            test_metrics[f"full-correct-{mname}"] = counters_correct[m]
            test_metrics[f"full-acc-{mname}"] = accuracies[m]
        # end for

        test_metrics[f"char-acc"] = np.mean(char_accuracies)
        test_metrics[f"frag-acc"] = np.mean(frag_accuracies)
        test_metrics[f"BLEU-4"] = np.mean(bleu_scores)

        self.logger.info(f"Test results:\n"
                         f"char-acc: {test_metrics['char-acc']}\n"
                         f"frag-acc: {test_metrics['frag-acc']}\n"
                         f"BLEU-4: {test_metrics['BLEU-4']}\n"
                         + "".join([f"{m} accuracy: {accuracies[m]:.4f} (count: {counters[m]})\n" for m in self.QUICK_METRICS]))

        # Save metrics
        IOUtils.dump(output_result_dir/"test-metrics.json", test_metrics, IOUtils.Format.jsonPretty)

        # Save predictions
        IOUtils.dump(output_result_dir/"predictions.txt", "".join([p+"\n" for p in predictions]), IOUtils.Format.txt)

        # Generate suggestions
        suggestions = self.generate_suggestions(lemma_names, lemma_qnames, data_indexes, predictions)
        IOUtils.dump(output_result_dir/"suggestions.txt", suggestions, IOUtils.Format.txt)
        return

    def get_modified_files(self, lemmas: List[Lemma], predictions: List[str], data_files_mgr: FilesManager, output_rel_path: str) -> NoReturn:
        data_files_mgr.clean_path(output_rel_path)

        data_indexes = list(set([l.data_index for l in lemmas]))

        for data_index in data_indexes:
            file_content: str = data_files_mgr.load_data([FilesManager.ORIGINAL_FILES, data_index], IOUtils.Format.txt)
            lemmas_idxes_this_file = [i for i, l in enumerate(lemmas) if l.data_index == data_index]
            for lemma_i in lemmas_idxes_this_file:
                lemma = lemmas[lemma_i]
                prediction = predictions[lemma_i]
                replace_from = (lemma.vernac_command_with_space() + " " + lemma.name).strip()
                replace_to = (lemma.vernac_command_with_space() + " " + prediction).strip()
                if replace_from not in file_content:  self.logger.warning(f"not found {replace_from} in {data_index}")
                file_content = file_content.replace(replace_from, replace_to, 1)
            # end for
            data_files_mgr.dump_data([output_rel_path, data_index], file_content, IOUtils.Format.txt)
        # end for

        return

    def combine_eval_results_trials(
            self,
            result_dirs: List[Path],
            output_result_dir: Path,
    ) -> NoReturn:
        test_metrics: List[dict] = list()

        for result_dir in result_dirs:
            test_metrics.append(IOUtils.load(result_dir/"test-metrics.json", IOUtils.Format.json))
        # end for

        test_metrics_keys = list(test_metrics[0].keys())

        combined_test_metrics: dict = {
            f"{s}-{k}": func([test_metric[k] for test_metric in test_metrics])
            for k in test_metrics_keys
            for s, func in Utils.SUMMARIES_FUNCS.items()
        }

        # Quick log results
        log_str = f"=-=-=-=-=- Averaged results from {len(test_metrics)} trials -=-=-=-=-=\n"
        log_str += f"char-acc: {combined_test_metrics['AVG-char-acc']:.4f}(±{combined_test_metrics['STDEV-char-acc']:.5f})\n"
        log_str += f"frag-acc: {combined_test_metrics['AVG-frag-acc']:.4f}(±{combined_test_metrics['STDEV-frag-acc']:.5f})\n"
        log_str += f"BLEU-4: {combined_test_metrics['AVG-BLEU-4']:.4f}(±{combined_test_metrics['STDEV-BLEU-4']:.5f})\n"
        for k in self.QUICK_METRICS:
            avg_accuracy = combined_test_metrics[f"AVG-full-acc-{k}"]
            stdev_accuracy = combined_test_metrics[f"STDEV-full-acc-{k}"]
            log_str += f"{k} full-acc: {avg_accuracy:.4f}(±{stdev_accuracy:.5f})\n"
        # end for
        self.logger.info(log_str)

        IOUtils.dump(output_result_dir/"test-metrics.json", combined_test_metrics, IOUtils.Format.jsonPretty)
        return

    def error_analyze(
            self,
            data_dir: Path,
            processed_data_dir: Path,
            result_dir: Path,
            output_report_dir: Path,
    ) -> NoReturn:
        # Load predictions
        predictions_results: List[List[Tuple[str, float]]] = IOUtils.load(result_dir/"predictions-results.json", IOUtils.Format.json)
        lemmas: List[Lemma] = IOUtils.dejsonfy(IOUtils.load(data_dir/"lemmas.json", IOUtils.Format.json), List[Lemma])

        inspection_reports: List[dict] = list()

        for i in range(len(lemmas)):
            truth = lemmas[i].name
            inspection_report = {
                "lemma": "".join([t.str_with_space() for t in lemmas[i].statement]),
                "truth": lemmas[i].name,
                "predictions": [
                    {
                        f"prediction #{pred_i}": pred,
                        "logprob": logprob,
                        "char-acc": self.acc([c for c in truth], [c for c in pred]),
                        "frag-acc": self.acc_unordered([f for f in truth.split("_")], [f for f in pred.split("_")]),
                        "BLEU-4": self.bleu([[c for c in lemmas[i].name]], [c for c in pred]),
                    } for pred_i, (pred, logprob) in enumerate(predictions_results[i])],
            }
            inspection_reports.append(inspection_report)
        # end for

        IOUtils.dump(output_report_dir/"inspections-reports.json", inspection_reports, IOUtils.Format.jsonNoSort)

        # Find valid predictions indexes
        valid_predictions_indexes = [i for i, predictions_result in enumerate(predictions_results) if self.is_legal_prediction(predictions_result[0][0])]

        # Random samples
        num_random_samples = 200
        random_samples: List[dict] = list()
        for pred_i in random.sample(valid_predictions_indexes, num_random_samples):
            random_samples.append(self.report_prediction_for_human_eval(predictions_results[pred_i], lemmas[pred_i]))
        # end for
        IOUtils.dump(output_report_dir/"random-samples.json", random_samples, IOUtils.Format.jsonNoSort)

        # Random files
        num_random_files = 5
        random_files_reports: List[dict] = list()
        data_indexes = set([lemmas[i].data_index for i in valid_predictions_indexes])
        for data_index in random.sample(data_indexes, num_random_files):
            lemmas_this_file_indexes = [i for i, l in enumerate(lemmas) if l.data_index == data_index and i in valid_predictions_indexes]
            random_files_reports.append({
                "project/file": data_index,
                "reports": [self.report_prediction_for_human_eval(predictions_results[i], lemmas[i]) for i in lemmas_this_file_indexes],
            })
        # end for
        IOUtils.dump(output_report_dir/"random-files-reports.json", random_files_reports, IOUtils.Format.jsonNoSort)

        # Best/Worst accuracy & confidence
        levels = ["files", "projects", "examples"]
        metrics = ["bleu", "frag-acc", "confidence"]
        levels_metrics_values: Dict[Tuple[str, str], Dict[str, list]] = dict()
        levels_reports: Dict[str, Dict[str, dict]] = dict()
        levels_keys: Dict[str, Set[str]] = dict()

        for level in levels:
            levels_reports[level] = dict()
            levels_keys[level] = set()
            for metric in metrics:  levels_metrics_values[(level, metric)] = collections.defaultdict(list)
        # end for

        for pred_i in valid_predictions_indexes:
            lemma = lemmas[pred_i]
            prediction, logprob = predictions_results[pred_i][0]
            truth = lemma.name

            bleu = self.bleu([[c for c in truth]], [c for c in prediction])
            frag_acc = self.acc_unordered([f for f in truth.split("_")], [f for f in prediction.split("_")])

            for level, key in {
                "files": lemma.data_index,
                "projects": lemma.data_index.split("/", 1)[0],
                "examples": lemma.data_index + ":" + lemma.qname,
            }.items():
                levels_keys[level].add(key)
                levels_metrics_values[(level, "bleu")][key].append(bleu)
                levels_metrics_values[(level, "frag-acc")][key].append(frag_acc)
                levels_metrics_values[(level, "confidence")][key].append(np.exp(logprob))
            # end for
        # end for

        for level in levels:
            for key in levels_keys[level]:
                levels_reports[level][key] = {
                    metric: np.mean(levels_metrics_values[(level, metric)][key])
                    for metric in metrics
                }
            # end for
        # end for

        for level, reports in levels_reports.items():
            num_pick = {
                "files": Macros.NUM_PICK_FILES,
                "projects": Macros.NUM_PICK_PRJS,
                "examples": Macros.NUM_PICK_EXAMPLES,
            }[level]
            if level != "examples":  IOUtils.dump(output_report_dir/f"{level}-reports.json", reports, IOUtils.Format.jsonPretty)
            for metric in metrics:
                IOUtils.dump(output_report_dir/f"{level}-{metric}-best.json", sorted(reports.items(), key=lambda kd: kd[1][metric], reverse=True)[:num_pick], IOUtils.Format.jsonPretty)
                IOUtils.dump(output_report_dir/f"{level}-{metric}-worst.json", sorted(reports.items(), key=lambda kd: kd[1][metric], reverse=False)[:num_pick], IOUtils.Format.jsonPretty)
            # end for
        # end for
        return

    @classmethod
    def report_prediction_for_human_eval(cls, predictions_logprobs: List[Tuple[str, float]], lemma: Lemma) -> Optional[dict]:
        truth = lemma.name
        prediction = predictions_logprobs[0][0]
        if not cls.is_legal_prediction(prediction):  return None
        return {
            "lemma_statement": lemma.statement_with_space(),
            "prediction": prediction,
            "project": lemma.data_index.split("/", 1)[0],
            "file": lemma.data_index.split("/", 1)[1],
            "truth": truth,
            "bleu": cls.bleu([[c for c in truth]], [c for c in prediction]),
            "frag-acc": cls.acc_unordered([f for f in truth.split("_")], [f for f in prediction.split("_")]),
        }

    @classmethod
    def generate_suggestions(cls, lemma_names: List[str], lemma_qnames: List[str], data_indexes: List[str], predictions: List[str]) -> str:
        indexes = list(range(len(lemma_names)))
        indexes.sort(key=lambda i: (data_indexes[i], lemma_qnames[i], lemma_names[i]))
        s = ""
        for i in indexes:
            if lemma_names[i] != predictions[i]:
                s += f"{data_indexes[i]}: {lemma_qnames[i]} -> {predictions[i]}\n"
            # end if
        return s

    def get_best_trial(self, result_dirs: List[Path]) -> Path:
        best_bleu = np.NINF
        best_result_dir = None
        for result_dir in result_dirs:
            metrics = IOUtils.load(result_dir/"test-metrics.json", IOUtils.Format.json)
            if metrics["BLEU-4"] > best_bleu:
                best_bleu = metrics["BLEU-4"]
                best_result_dir = result_dir
            # end if
        # end for
        return best_result_dir

    @classmethod
    def is_legal_prediction(self, prediction: str) -> bool:
        return set("().,#").isdisjoint(set(prediction))

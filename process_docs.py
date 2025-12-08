import json
from zensols.calamr import FlowGraphResult, ApplicationFactory
from pyg_export import PyTorchGeometricExporter as PyGExport
from pathlib import Path
import torch
import config
from typing import Dict, Any, Tuple


def _load_labels_as_map(dataset: str, subset: str) -> Dict[str, int]:
    with open(f"calamr_input/{dataset}/{subset}/final_decisions.json") as f:
        final_decisions = json.load(f)
    return {decision["id"]: decision["final_decision"] for decision in final_decisions}


def _get_corpus(dataset: str, subset: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    with open(f"calamr_input/{dataset}/{subset}/calamr_data.json") as f:
        corpus = json.load(f)[:5]

    labels = _load_labels_as_map(dataset=dataset, subset=subset)

    return corpus, labels


def process_docs(dataset: str = "medhallu", subset: str = "labeled"):
    resources = ApplicationFactory.get_resources()
    pyg_export = PyGExport()

    corpus, labels = _get_corpus(dataset=dataset, subset=subset)

    # make output paths
    output_path = Path(f"{config.RESULTS_DIR}/medhallu/v8/{subset}")
    output_path.mkdir(parents=True, exist_ok=True)

    # make pyg output paths
    pyg_output_path = output_path / "pyg"
    pyg_output_path.mkdir(parents=True, exist_ok=True)

    # access an ad hoc corpus as defined above with the list of dictionaries above
    with resources.adhoc(corpus) as r:
        # list the keys in the corpus, each of which is available as a document or
        # alignment as flow metrics/data
        docs = list(r.documents.keys())

        for doc_id in docs:
            try:
                flow: FlowGraphResult = r.alignments[doc_id]
                pyg_data = pyg_export.export_alignment_graph(flow)
                label = labels[doc_id]
                pyg_data.y = torch.tensor([label], dtype=torch.long)

                pyg_path = Path(output_path) / "pyg" / f"{doc_id}.pt"
                torch.save(pyg_data, pyg_path)
            except Exception:
                print(f"Error processing document {doc_id}")


if __name__ == "__main__":
    process_docs(dataset="medhallu", subset="labeled")
    process_docs(dataset="medhallu", subset="artificial")

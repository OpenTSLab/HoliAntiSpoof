from tqdm import tqdm
import sed_eval


def calculate_segment_f1(
    gt_segments: dict[str, list[list[float]]],
    pred_segments: dict[str, list[list[float]]],
    resolution: float = 0.2,
):
    evaluator = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=["fake"],
        time_resolution=resolution,
    )

    assert gt_segments.keys() == pred_segments.keys()
    for audio_id in tqdm(gt_segments):
        ref_data, pred_data = [], []
        gt_segments_ = gt_segments[audio_id]
        pred_segments_ = pred_segments[audio_id]
        for segment in gt_segments_:
            ref_data.append({
                "event_label": "fake",
                "event_onset": segment[0],
                "event_offset": segment[1],
                "file_name": audio_id,
            })
        for segment in pred_segments_:
            pred_data.append({
                "event_label": "fake",
                "event_onset": segment[0],
                "event_offset": segment[1],
                "file_name": audio_id,
            })

        evaluator.evaluate(ref_data, pred_data)

    metrics = evaluator.results_overall_metrics()

    return metrics

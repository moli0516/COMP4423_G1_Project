"""Pair-level false-change alarms on independently verified no-change pairs."""


class NoChangeEvaluator:
    """Eligibility is ground-truth metadata, never inferred from predictions."""

    STATUSES = ("added", "removed", "moved", "unchanged")

    def evaluate(self, results: dict, eligible_pair_ids: list[str]) -> dict:
        """Accept {pair_id: [event, ...]} and unique eligible IDs.

        Events require only a known status. Geometry and event localization are
        EventEvaluator responsibilities. Extra noneligible pairs are ignored.
        An explicit empty result is not a false alarm, but is not proof of
        successful detection; missing eligible results are errors.
        """
        if not isinstance(results, dict):
            raise TypeError("Results must be a dictionary keyed by pair ID")
        if not isinstance(eligible_pair_ids, list):
            raise TypeError("Eligible pair IDs must be a list")
        if not eligible_pair_ids:
            raise ValueError("At least one eligible no-change pair is required")
        if any(not isinstance(p, str) or not p for p in eligible_pair_ids):
            raise ValueError("Eligible pair IDs must be nonempty strings")
        if len(set(eligible_pair_ids)) != len(eligible_pair_ids):
            raise ValueError("Duplicate eligible pair ID")
        missing = set(eligible_pair_ids) - results.keys()
        if missing:
            raise ValueError(f"Missing eligible results: {sorted(missing)}")
        alarm_ids = []
        for pair_id in sorted(eligible_pair_ids):
            events = results[pair_id]
            if not isinstance(events, list):
                raise TypeError("Each eligible result must be a list of events")
            alarm = False
            for event in events:
                if not isinstance(event, dict):
                    raise TypeError("Each event must be a dictionary")
                if event.get("status") not in self.STATUSES:
                    raise ValueError("Missing or unknown event status")
                if event["status"] != "unchanged":
                    alarm = True
            if alarm:
                alarm_ids.append(pair_id)
        return {
            "protocol_version": "independent-no-change-v1",
            "eligible_pair_count": len(eligible_pair_ids),
            "false_alarm_pair_count": len(alarm_ids),
            "false_alarm_rate": len(alarm_ids) / len(eligible_pair_ids),
            "false_alarm_pair_ids": alarm_ids,
        }

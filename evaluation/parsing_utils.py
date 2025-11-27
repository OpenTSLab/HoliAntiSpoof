import re
import json


class SpoofingParser:
    def __init__(self, data_format: str):
        assert data_format in ("json", "cot"), f"data format {data_format} not recognized"
        self.data_format = data_format
        if self.data_format == "cot":
            # Precompile regex to speed up repeated matches and reduce backtracking
            self.real_fake_regex = re.compile(r"The utterance is ([^.]+)\.", re.DOTALL)
            self.reasoning_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
            self.semantic_influence_regex = re.compile(
                r"The spoofing operation may result in the following influence: (.*)", re.DOTALL
            )
            self.spoof_method_regex = re.compile(r"This indicates the spoof method is ([^.]+)\.", re.DOTALL)
            # Regex for single fake region: "The fake region is: xxx-xxx seconds."
            self.fake_region_regex = re.compile(r'The fake region is: ([0-9.]+-[0-9.]+) seconds\.', re.DOTALL)
            # Regex for multiple fake regions: "The fake regions are: xxx-xxx seconds, yyy-yyy seconds."
            self.fake_regions_regex = re.compile(
                r'The fake regions are: ([0-9.]+-[0-9.]+ seconds(?:, [0-9.]+-[0-9.]+ seconds)*)\.', re.DOTALL
            )
            # Regex to extract all time ranges (xxx-xxx format) from text
            self.time_range_regex = re.compile(r'([0-9.]+-[0-9.]+)')
            self.real_fake_transform = {"real": "real", "a spoof": "fake"}

    def __call__(self, text):
        if self.data_format == "json":
            format_success = True
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                format_success = False
                real_or_fake, spoof_method, fake_region, semantic_influence = None, None, None, None

            if format_success:
                real_or_fake = data.get("real_or_fake", None)
                spoof_method = data.get("spoof_method", None)
                fake_region = data.get("fake_region", None)
                semantic_influence = data.get("semantic_influence", None)

        elif self.data_format == "cot":

            format_success = validate_cot_format(text)

            answer_start_idx = text.find("</think>") + len("</think>")
            answer = text[answer_start_idx:]
            search_res = self.real_fake_regex.search(answer)
            if search_res:
                real_or_fake = search_res.group(1)
                semantic_influence = self.semantic_influence_regex.search(answer)
                if semantic_influence:
                    semantic_influence = semantic_influence.group(1)
            else:
                real_or_fake = None
                semantic_influence = None

            search_res = self.reasoning_regex.search(text)

            if search_res:
                reasoning_content = search_res.group(1)
                spoof_method = self.spoof_method_regex.search(reasoning_content)
                if spoof_method:
                    spoof_method = spoof_method.group(1)

                # Extract fake_region from reasoning_content
                fake_region = None
                # Try single fake region first

                if "The entire utterance is manipulated." in reasoning_content:
                    fake_region = "all"
                else:
                    region_match = self.fake_region_regex.search(reasoning_content)
                    if region_match:
                        fake_region_text = [region_match.group(1)]
                    else:
                        # Try multiple fake regions
                        regions_match = self.fake_regions_regex.search(reasoning_content)
                        if regions_match:
                            # Extract all time ranges from the matched text
                            matched_text = regions_match.group(0)
                            fake_region_text = self.time_range_regex.findall(matched_text)
                        else:
                            fake_region_text = None

                    if fake_region_text:
                        fake_region = []
                        for segment in fake_region_text:
                            start, end = segment.split("-")
                            fake_region.append([float(start), float(end)])

            else:
                spoof_method, fake_region = None, None

            if real_or_fake:
                real_or_fake = self.real_fake_transform.get(real_or_fake, None)

        if not self.validate_fake_region(fake_region):
            fake_region = None

        return {
            "real_or_fake": real_or_fake,
            "semantic_influence": semantic_influence,
            "spoof_method": spoof_method,
            "fake_region": fake_region,
            "format_success": format_success,
        }

    def validate_fake_region(self, fake_region: list[list[float]] | str | None) -> bool:
        if fake_region is None:
            return True
        elif isinstance(fake_region, str):
            return fake_region == "all"
        elif isinstance(fake_region, list):
            validate = True
            for region in fake_region:
                if not isinstance(region, list):
                    validate = False
                else:
                    if len(region) != 2:
                        validate = False
                    else:
                        if not (isinstance(region[0], (int, float)) and isinstance(region[1], (int, float))):
                            validate = False
                        else:
                            if region[0] >= region[1]:
                                validate = False
                if not validate:
                    break
            return validate
        return False


def validate_cot_format(text: str) -> bool:
    """Validate if text matches the CoT format requirements.
    
    Format requirements:
    <think>
    xxxxThis indicates the spoof method is xxx
    The transcription of this utterance is: "xxxx".
    </think>

    The utterance is xxxx
    
    Args:
        text: The text to validate
        
    Returns:
        bool: Returns True if format is correct, False otherwise
    """
    pattern = re.compile(
        r'<think>.*?The transcription of this utterance is: "[^"]*".*?</think>\n\nThe utterance is (?:a spoof|real)\..*?',
        re.DOTALL
    )
    return bool(pattern.search(text))


def init_parser(data: list[dict] | None = None, data_format: str | None = None) -> SpoofingParser:
    if data_format is None:
        if data is None:
            raise ValueError("data_format and data cannot be None at the same time")
        try:
            json.loads(data[0]['ref'])
        except json.JSONDecodeError:
            data_format = "cot"
        else:
            data_format = "json"

    return SpoofingParser(data_format)

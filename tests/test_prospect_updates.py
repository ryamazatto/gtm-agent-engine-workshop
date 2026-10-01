import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


class ProspectUpdateTest(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()
        self.original_tech_stack = list(data_service.PROSPECTS["LEAD-39002"]["tech_stack"])

    def tearDown(self):
        data_service.PROSPECTS["LEAD-39002"]["tech_stack"] = self.original_tech_stack
        data_service._PROFILES.clear()

    def test_update_rebuilds_profile_and_scoring_uses_current_tech_stack(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertNotIn("Terraform", profile["prospect_profile"]["tech_stack"])

        update = data_service.update_prospect_info("LEAD-39002", "Terraform")
        self.assertEqual(update["updated"], True)
        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002")[-1], "Terraform")
        duplicate = data_service.update_prospect_info("LEAD-39002", "Terraform")
        self.assertEqual(duplicate["tech_stack"].count("Terraform"), 1)

        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

        captured = {}

        def invoke(messages):
            captured["user"] = messages[1]["content"]
            return SimpleNamespace(
                model_dump=lambda: {
                    "score": 80,
                    "justification": "Terraform is present.",
                }
            )

        with patch.object(gtm_agent, "_scoring_llm", SimpleNamespace(invoke=invoke)):
            gtm_agent.score_prospect.invoke({
                "prospect_profile": rebuilt["prospect_profile"],
                "offering": gtm_agent.lookup_offering.invoke({"offering_id": "OFFER-10004"})["offering"],
            })

        self.assertIn('"Terraform"', captured["user"])
        self.assertIn('"present": [\n    "Terraform"', captured["user"])
        self.assertIn('"missing": []', captured["user"])


if __name__ == "__main__":
    unittest.main()

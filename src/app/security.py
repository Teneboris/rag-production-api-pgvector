"""
Security Layer
Input sanitization, personally identifiable information (PII) detection/masking, output validation
"""

from typing import Optional

from langsmith import traceable

# === Input Sanitization ===

class InputSanitizer:
    """
    Sanitize user input before it reaches the LLM.
    Detects prompts injection patterns and cleans dangerous content.
    
    """
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions\s*:",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
        r"reveal\s+(your|the)\s+(system|instructions|prompt)",
        r"you\s+are\s+now\s+(DAN|jailbroken)",
    ]

    def __init__(self):
        self.patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INJECTION_PATTERNS
        ]

    @traceable(name="security_check")
    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Check whether the input contains known prompt-injection patterns.

        Returns:
            tuple:
                is_safe: True if no suspicious pattern is found.
                rejection_reason: Reason when blocked, otherwise None.
        """

        if not isinstance(text, str):
            return False, "Blocked: input must be a string"

        if not text.strip():
            return False, "Blocked: input cannot be empty"

        for pattern in self.patterns:
            if pattern.search(text):
                return (
                    False,
                    f"Blocked: potential prompt injection detected "
                    f"({pattern.pattern})",
                )

        return True, None

    def clean(self, text: str) -> str:
        """
        Remove or neutralize some potentially dangerous delimiters.
        """

        # Remove repeated delimiters such as --- or ===
        text = re.sub(r"-{3,}", "", text)
        text = re.sub(r"={3,}", "", text)

        # Neutralize template-style delimiters
        text = text.replace("{{", "{ {")
        text = text.replace("}}", "} }")

        # Remove leading/trailing whitespace
        return text.strip()
import re


class PIIDetector:
    """
    Detect and mask personally identifiable information (PII).

    Works on both input (before LLM)
    and output (before client).
    """

    PATTERN = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),

        "phone": re.compile(
            r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b"
        ),

        "ssn": re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b"
        ),

        "credit_card": re.compile(
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        ),
    }

    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
    }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII types present in text."""

        found = {}

        for pii_type, pattern in self.PATTERN.items():
            matches = pattern.findall(text)

            if matches:
                found[pii_type] = matches

        return found

    def mask(self, text: str) -> str:
        """Replace all detected PII with redaction markers."""

        masked = text

        for pii_type, pattern in self.PATTERN.items():
            masked = pattern.sub(
                self.MASK_MAP[pii_type],
                masked
            )

        return masked
    
# output validator

class OutputValidator:
    """
    Validate LLM output before returning to the client.
    Catches PII leakage and harmful content in response
    """
    HARMFUL_PATTERN = [
        re.compile(
            r"here(?:['´]s| is)\s+how\s+to\s+(hack|steal|attack)",
            re.I,
        ),
        re.compile(r"password\s+is\s+", re.I),
        re.compile(r"api[_\s]?key\s*[:=]", re.I),
    ]
    
    def __init__(self):
        self.pii_detector = PIIDetector()
    
    def validate(self, output: str) -> tuple[str, list[str]]:
        """
        Validate and clean output.
        Returns: (cleaned_ouput, list_of_warnings)
        """
        warnings = []
        
        # Check for PII leakage in output
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")
    
        # Check for harmful content
        for pattern in self.HARMFUL_PATTERN:
            if pattern.search(output):
                output = "Response blocked: potentially harmful content"
                warnings.append(f"Harmful content blocked")
                break
        return output, warnings

class SecurityPipeline:
    """
    Full security pipeline that processes input and output.
    this is the single class to wire into the API
    """

    def __init__(self,):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.out_validator = OutputValidator()

    @traceable(name="security_check_input")
    def check_input(self, text:str) -> tuple[bool, str, list[str]]:
        """
        Process input through security checks.
        Returns: (is_allowed, cleaned_text, security_notes)
        """

        notes = []
        
        # Step 1: Check for injection
        is_safe, reason = self.sanitizer.check(text)
        if not is_safe:
            return False, "", [reason]

        # Step 2: Clean Input
        cleaned = self.sanitizer.clean(text)

        # Step 3: Mask PII before it reaches the LLM
        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(text)
            notes.append(f'Input PII masked: {list(pii_found.keys())}')
            
        return True, cleaned, notes
    
    @traceable(name="security_check_output")
    def check_output(self, text:str) -> tuple[str, list[str]]:
        """
        Validate output before returning to client.
        Return: (cleaned_ouput, warnings)
        """
        return self.out_validator.validate(text)
 
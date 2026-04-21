"""
Frame processing modules
"""
from .config import FrameConfig
from .synchronizer import FrameSynchronizer
from .channel_estimator import ChannelEstimator
from .symbol_detector import SymbolDetector
from .data_manager import DataManager
from .detector_interfaces import (
    DetectorFrameInputs,
    DetectionRunResult,
    DetectionStep,
    ContextPair,
    DecisionFeedbackPromptBuilder,
    AbstractDetectorBackend,
    CoherentMMSEDetector,
    MMSEDecisionFeedbackDetector,
    DefinedDetectorAdapter,
)
from .receiver_experiment import (
    PayloadCase,
    PayloadDetectionResult,
    DetectorExperimentSummary,
    build_payload_cases_from_phase_result,
    run_detector_on_cases,
    make_detector,
)

__all__ = [
    'FrameConfig',
    'FrameSynchronizer',
    'ChannelEstimator',
    'SymbolDetector',
    'DataManager',
    'DetectorFrameInputs',
    'DetectionRunResult',
    'DetectionStep',
    'ContextPair',
    'DecisionFeedbackPromptBuilder',
    'AbstractDetectorBackend',
    'CoherentMMSEDetector',
    'MMSEDecisionFeedbackDetector',
    'DefinedDetectorAdapter',
    'PayloadCase',
    'PayloadDetectionResult',
    'DetectorExperimentSummary',
    'build_payload_cases_from_phase_result',
    'run_detector_on_cases',
    'make_detector',
]

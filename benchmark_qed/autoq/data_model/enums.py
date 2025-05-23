from enum import Enum


class QuestionType(str, Enum):
    """Enum for question types"""

    DATA_LOCAL = "data_local"
    DATA_GLOBAL = "data_global"
    ACTIVITY_LOCAL = "activity_local"
    ACTIVITY_GLOBAL = "activity_global"

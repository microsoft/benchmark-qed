# Copyright (c) 2025 Microsoft Corporation.
"""Compare app for displaying results of the answer evaluation."""

from collections.abc import Callable
from pathlib import Path, PurePath
from typing import ClassVar, TypeAlias

import pandas as pd
from textual import on
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Header, Label, Markdown, Select, Static

CSSPathType: TypeAlias = str | PurePath | list[str | PurePath]


class ReasoningWidget(Widget):
    """Widget to display the reasoning behind the scores."""

    reasoning: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Static("Reasoning:")
        yield Markdown(id="reasoning")

    def watch_reasoning(self, new_reasoning: str) -> None:
        """Update the reasoning label when the reasoning changes."""
        self.query_one("#reasoning", Markdown).update(f"{new_reasoning}")


class AnswerPairWidget(Widget):
    """Widget to display the answers for each condition."""

    question: reactive[str] = reactive("")
    condition_1: reactive[str] = reactive("")
    condition_2: reactive[str] = reactive("")
    answer_1: reactive[str] = reactive("")
    answer_2: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Label(id="condition_1")
        yield Label(id="condition_2")
        yield Markdown(id="answer_1")
        yield Markdown(id="answer_2")

    def watch_condition_1(self, new_name: str) -> None:
        """Update the condition 1 name label when the condition 1 name changes."""
        self.query_one("#condition_1", Label).update(f"{new_name}")

    def watch_condition_2(self, new_name: str) -> None:
        """Update the condition 2 name label when the condition 2 name changes."""
        self.query_one("#condition_2", Label).update(f"{new_name}")

    def watch_answer_1(self, new_answer: str) -> None:
        """Update the answer 1 label when the answer 1 changes."""
        self.query_one("#answer_1", Markdown).update(f"{new_answer}")

    def watch_answer_2(self, new_answer: str) -> None:
        """Update the answer 2 label when the answer 2 changes."""
        self.query_one("#answer_2", Markdown).update(f"{new_answer}")


class SummaryInfoWidget(Widget):
    """Widget to display summary information about the conditions."""

    condition_1_name: reactive[str] = reactive("")
    condition_2_name: reactive[str] = reactive("")
    condition_1_mean: reactive[float] = reactive(0.0)
    condition_2_mean: reactive[float] = reactive(0.0)
    z_value: reactive[float] = reactive(0.0)
    trial: reactive[int] = reactive(0)
    p_value: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Label(id="condition_1_name")
        yield Label(id="condition_1_mean")
        yield Label(id="z_value")

        yield Label(id="condition_2_name")
        yield Label(id="condition_2_mean")
        yield Label(id="p_value")

    def watch_z_value(self, new_z_value: float) -> None:
        """Update the z-value label when the z-value changes."""
        self.query_one("#z_value", Label).update(f"Z-value: {new_z_value}")

    def watch_p_value(self, new_p_value: float) -> None:
        """Update the p-value label when the p-value changes."""
        self.query_one("#p_value", Label).update(f"p-value: {new_p_value}")

    def watch_condition_1_name(self, new_name: str) -> None:
        """Update the condition 1 name label when the condition 1 name changes."""
        self.query_one("#condition_1_name", Label).update(f"Condition 1: {new_name}")

    def watch_condition_1_mean(self, new_mean: float) -> None:
        """Update the condition 1 mean label when the condition 1 mean changes."""
        self.query_one("#condition_1_mean", Label).update(f"Mean: {new_mean}")

    def watch_condition_2_name(self, new_name: str) -> None:
        """Update the condition 2 name label when the condition 2 name changes."""
        self.query_one("#condition_2_name", Label).update(f"Condition 2: {new_name}")

    def watch_condition_2_mean(self, new_mean: float) -> None:
        """Update the condition 2 mean label when the condition 2 mean changes."""
        self.query_one("#condition_2_mean", Label).update(f"Mean: {new_mean}")


class SelectorWidget(Widget):
    """Widget to select the criteria, question, and trial."""

    def __init__(
        self,
        criteria: list[str],
        questions: list[str],
        trials: list[int],
        on_criteria_selected: Callable,
        on_question_selected: Callable,
        on_trial_selected: Callable,
    ) -> None:
        super().__init__()
        self._criteria = criteria
        self._questions = questions
        self._trials = trials
        self._on_criteria_selected = on_criteria_selected
        self._on_question_selected = on_question_selected
        self._on_trial_selected = on_trial_selected

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Label("Criteria:")
        yield Label("Question:", id="question-label")
        yield Label("Trial")

        yield Select.from_values(
            self._criteria, allow_blank=False, id="select-criteria"
        )
        yield Select.from_values(
            self._questions, allow_blank=False, id="select-question"
        )
        yield Select.from_values(self._trials, allow_blank=False, id="select-trial")

    @on(Select.Changed, selector="#select-criteria")
    def criteria_changed(self, event: Select.Changed) -> None:
        """Handle criteria selection change."""
        self._on_criteria_selected(str(event.value))

    @on(Select.Changed, selector="#select-question")
    def question_changed(self, event: Select.Changed) -> None:
        """Handle question selection change."""
        self._on_question_selected(str(event.value))

    @on(Select.Changed, selector="#select-trial")
    def trial_changed(self, event: Select.Changed) -> None:
        """Handle trial selection change."""
        self._on_trial_selected(int(str(event.value)))


class CompareApp(App[None]):
    """A Textual app to manage stopwatches."""

    CSS_PATH: ClassVar[CSSPathType | None] = "compare_app.tcss"  # type: ignore

    BINDINGS: ClassVar = [
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self._all_results = pd.read_csv(output_path / "pairwise_scores.csv")
        self._significance = pd.read_csv(output_path / "pairwise_scores_p_value.csv")
        self._criteria = self._all_results["criteria"].unique()[0]
        self._question = self._all_results["question"].unique()[0]
        self._trial = self._all_results["trial"].unique()[0]
        self._index = 0

    @property
    def selector_widget(self) -> SelectorWidget:
        """Get the selector widget."""
        return self.query_one(SelectorWidget)

    @property
    def summary_widget(self) -> SummaryInfoWidget:
        """Get the summary widget."""
        return self.query_one(SummaryInfoWidget)

    @property
    def answer_pair_widget(self) -> AnswerPairWidget:
        """Get the answer pair widget."""
        return self.query_one(AnswerPairWidget)

    @property
    def reasoning_widget(self) -> ReasoningWidget:
        """Get the reasoning widget."""
        return self.query_one(ReasoningWidget)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield SelectorWidget(
            list(self._all_results["criteria"].unique()),
            list(self._all_results["question"].unique()),
            list(self._all_results["trial"].unique()),
            lambda x: self.on_criteria_selected(x),
            lambda x: self.on_question_selected(x),
            lambda x: self.on_trial_selected(x),
        )
        yield SummaryInfoWidget()
        yield AnswerPairWidget()
        yield ReasoningWidget()
        yield Footer()

    def on_criteria_selected(self, criteria: str) -> None:
        """Handle criteria selection change."""
        self._criteria = criteria
        self.update_screen()

    def on_question_selected(self, question: str) -> None:
        """Handle question selection change."""
        self._question = question
        self.update_screen()

    def on_trial_selected(self, trial: int) -> None:
        """Handle trial selection change."""
        self._trial = trial
        self.update_screen()

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def update_screen(self) -> None:
        """Update the screen with the current selection."""
        current_row = self._all_results[
            (self._all_results["criteria"] == self._criteria)
            & (self._all_results["question"] == self._question)
            & (self._all_results["trial"] == self._trial)
        ].reset_index()
        if len(current_row) == 0:
            return
        current_row = current_row.iloc[0]
        significance_row = self._significance[
            self._significance["criteria"] == self._criteria
        ].iloc[0]
        self.summary_widget.condition_1_name = current_row["answer_1_name"]
        self.summary_widget.condition_2_name = current_row["answer_2_name"]
        self.summary_widget.condition_1_mean = significance_row[
            "base_mean"
            if current_row["answer_1_name"] == current_row["base_name"]
            else "other_mean"
        ]
        self.summary_widget.condition_2_mean = significance_row[
            "base_mean"
            if current_row["answer_2_name"] == current_row["base_name"]
            else "other_mean"
        ]
        self.summary_widget.z_value = significance_row["statistic"]
        self.summary_widget.p_value = significance_row["formatted_corrected_p_value"]

        self.answer_pair_widget.question = current_row["question"]
        self.answer_pair_widget.condition_1 = current_row["answer_1_name"]
        self.answer_pair_widget.condition_2 = current_row["answer_2_name"]
        self.answer_pair_widget.answer_1 = current_row["answer_1"]
        self.answer_pair_widget.answer_2 = current_row["answer_2"]

        self.reasoning_widget.reasoning = current_row["reasoning"]

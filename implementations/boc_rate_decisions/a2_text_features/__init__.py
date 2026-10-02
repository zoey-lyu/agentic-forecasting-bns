"""A2 — statement text as predictor features, by extraction rather than prediction.

An LLM reads each Bank of Canada statement and returns a structured stance
record; a logistic regression fitted at the origin decides what that record is
worth. Splitting the work this way is the point: predicting probabilities is
what the model is measurably bad at here, describing a document is what it is
good at, and only separating them shows whether the text carries anything.

Modules
-------
``stance``
    The extraction schema, prompt, and the verbatim-quote check that catches a
    fabricated record.
``extract_stance_table``
    Runs the extraction once over every cached statement; resumable.
``features``
    ``StanceTable`` — cutoff-aware feature rows, plus the seeded shuffle the
    control arm uses.
``text_logistic``
    ``BoCTextLogisticPredictor`` — macro features with or without the stance
    block, one class for all three arms so only the features differ.
``run_a2``
    Scores the arms, checks the ablation reproduces the macro baseline exactly,
    and runs the permutation null.
"""

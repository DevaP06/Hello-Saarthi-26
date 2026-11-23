ALTER TABLE saarthi_quiz_questions ADD COLUMN IF NOT EXISTS revised_at TIMESTAMP WITH TIME ZONE;

CREATE TABLE IF NOT EXISTS saarthi_quiz_practice (
    user_id INTEGER NOT NULL REFERENCES saarthi_users(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES saarthi_quiz_questions(id) ON DELETE CASCADE,
    attempt_id INTEGER NOT NULL REFERENCES saarthi_quiz_attempts(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    correct_streak INTEGER NOT NULL DEFAULT 0,
    selected_index INTEGER NOT NULL,
    next_review_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (user_id, question_id)
);

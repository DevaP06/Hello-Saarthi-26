-- Code Lab problems can belong to a course and one of its lessons (Classwork topic).
-- course_id NULL = a practice problem every user sees (the seeded ones).
ALTER TABLE saarthi_code_problems ADD COLUMN IF NOT EXISTS course_id INTEGER REFERENCES saarthi_courses(id) ON DELETE CASCADE;
ALTER TABLE saarthi_code_problems ADD COLUMN IF NOT EXISTS topic VARCHAR(128);
CREATE INDEX IF NOT EXISTS ix_saarthi_code_problems_course_id ON saarthi_code_problems (course_id);

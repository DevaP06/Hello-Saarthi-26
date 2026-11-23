ALTER TABLE saarthi_courses ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE saarthi_courses ADD COLUMN IF NOT EXISTS syllabus_name VARCHAR(255);
ALTER TABLE saarthi_courses ADD COLUMN IF NOT EXISTS syllabus_pdf BYTEA;

CREATE TABLE IF NOT EXISTS saarthi_module_prerequisites (
    module_id INTEGER NOT NULL REFERENCES saarthi_course_modules(id) ON DELETE CASCADE,
    prerequisite_id INTEGER NOT NULL REFERENCES saarthi_course_modules(id) ON DELETE CASCADE,
    PRIMARY KEY (module_id, prerequisite_id),
    CHECK (module_id <> prerequisite_id)
);

CREATE INDEX IF NOT EXISTS ix_saarthi_module_prerequisites_prerequisite_id ON saarthi_module_prerequisites (prerequisite_id);

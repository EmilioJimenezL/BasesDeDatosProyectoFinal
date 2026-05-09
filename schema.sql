-- LSI Document Retrieval System — MySQL Schema
-- Database: docbase

CREATE DATABASE IF NOT EXISTS docbase
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_spanish_ci;

USE docbase;

CREATE TABLE IF NOT EXISTS DOCUMENT (
    id       INT          AUTO_INCREMENT PRIMARY KEY,
    url      VARCHAR(512),
    title    VARCHAR(255),
    author   VARCHAR(255),
    doc_date DATE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

CREATE TABLE IF NOT EXISTS TERM (
    id   INT          AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

CREATE TABLE IF NOT EXISTS WORD (
    id      INT          AUTO_INCREMENT PRIMARY KEY,
    term_id INT          NOT NULL,
    word    VARCHAR(100),
    CONSTRAINT fk_word_term
        FOREIGN KEY (term_id) REFERENCES TERM(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

CREATE TABLE IF NOT EXISTS HAS (
    document_id INT   NOT NULL,
    term_id     INT   NOT NULL,
    frequency   FLOAT NOT NULL DEFAULT 0,
    PRIMARY KEY (document_id, term_id),
    CONSTRAINT fk_has_document
        FOREIGN KEY (document_id) REFERENCES DOCUMENT(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_has_term
        FOREIGN KEY (term_id) REFERENCES TERM(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

CREATE TABLE IF NOT EXISTS QUERY (
    id          INT          AUTO_INCREMENT PRIMARY KEY,
    label       VARCHAR(100),
    document_id INT,
    CONSTRAINT fk_query_document
        FOREIGN KEY (document_id) REFERENCES DOCUMENT(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

CREATE TABLE IF NOT EXISTS SVD_MATRIX (
    id          INT   AUTO_INCREMENT PRIMARY KEY,
    term_id     INT   NOT NULL,
    document_id INT   NOT NULL,
    t_value     FLOAT,
    s_value     FLOAT,
    d_value     FLOAT,
    k_rank      INT,
    CONSTRAINT fk_svd_term
        FOREIGN KEY (term_id) REFERENCES TERM(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_svd_document
        FOREIGN KEY (document_id) REFERENCES DOCUMENT(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

CREATE TABLE IF NOT EXISTS STOP_WORD (
    id   INT          AUTO_INCREMENT PRIMARY KEY,
    word VARCHAR(100) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

CREATE TABLE IF NOT EXISTS SUFFIX (
    id          INT         AUTO_INCREMENT PRIMARY KEY,
    suffix      VARCHAR(50),
    replacement VARCHAR(50) DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_spanish_ci;

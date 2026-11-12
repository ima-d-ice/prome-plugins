"""Generate deterministic synthetic code fixtures for multiple languages."""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def make_python() -> None:
    code = '''"""Prome Code Extraction Module."""

import math
from typing import List

class Calculator:
    """Performs numerical operations."""

    def __init__(self, precision: int = 2) -> None:
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)

def compute_hypotenuse(a: float, b: float) -> float:
    """Compute hypotenuse of right triangle."""
    return math.sqrt(a ** 2 + b ** 2)
'''
    (FIXTURES / "sample.py").write_text(code.strip())


def make_javascript() -> None:
    code = '''import { logger } from './utils.js';

class TaskRunner {
    constructor(concurrency) {
        this.concurrency = concurrency;
    }

    runTask(taskId) {
        logger.info(`Running task ${taskId}`);
        return true;
    }
}

function processBatch(tasks) {
    return tasks.map(t => t.id);
}
export { TaskRunner, processBatch };
'''
    (FIXTURES / "sample.js").write_text(code.strip())


def make_typescript() -> None:
    code = '''interface UserProfile {
    id: number;
    username: string;
    isActive: boolean;
}

class UserService {
    private users: UserProfile[] = [];

    findUser(id: number): UserProfile | undefined {
        return this.users.find(u => u.id === id);
    }
}

function formatGreeting(user: UserProfile): string {
    return `Hello, ${user.username}!`;
}
'''
    (FIXTURES / "sample.ts").write_text(code.strip())


def make_java() -> None:
    code = '''package com.prome.parser;

import java.util.List;
import java.util.ArrayList;

public class DataPipeline {
    private List<String> records = new ArrayList<>();

    public void addRecord(String record) {
        this.records.add(record);
    }

    public int getRecordCount() {
        return this.records.size();
    }
}
'''
    (FIXTURES / "sample.java").write_text(code.strip())


def make_c() -> None:
    code = '''#include <stdio.h>
#include <stdlib.h>

int multiply(int a, int b) {
    return a * b;
}

int main(int argc, char** argv) {
    printf("Result: %d\\n", multiply(6, 7));
    return 0;
}
'''
    (FIXTURES / "sample.c").write_text(code.strip())


def make_cpp() -> None:
    code = '''#include <iostream>
#include <vector>

class VectorBuffer {
public:
    void push(int value) {
        items.push_back(value);
    }

    size_t size() const {
        return items.size();
    }
private:
    std::vector<int> items;
};
'''
    (FIXTURES / "sample.cpp").write_text(code.strip())


def make_go() -> None:
    code = '''package service

import (
    "fmt"
    "time"
)

type Worker struct {
    ID int
}

func (w *Worker) Start() {
    fmt.Printf("Worker %d starting at %v\\n", w.ID, time.Now())
}

func NewWorker(id int) *Worker {
    return &Worker{ID: id}
}
'''
    (FIXTURES / "sample.go").write_text(code.strip())


def make_empty() -> None:
    (FIXTURES / "empty.py").write_text("")


def make_malformed() -> None:
    (FIXTURES / "malformed.py").write_text("def broken_syntax(:\n   return")


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_python()
    make_javascript()
    make_typescript()
    make_java()
    make_c()
    make_cpp()
    make_go()
    make_empty()
    make_malformed()
    print(f"Generated code fixtures in {FIXTURES}")


if __name__ == "__main__":
    main()

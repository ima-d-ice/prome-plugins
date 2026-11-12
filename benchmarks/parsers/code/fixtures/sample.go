package service

import (
    "fmt"
    "time"
)

type Worker struct {
    ID int
}

func (w *Worker) Start() {
    fmt.Printf("Worker %d starting at %v\n", w.ID, time.Now())
}

func NewWorker(id int) *Worker {
    return &Worker{ID: id}
}
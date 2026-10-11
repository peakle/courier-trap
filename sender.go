package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

// Sender отправляет события на бэкенд асинхронно, с ретраями и бэкоффом.
// Если url пустой — печатает события в stdout (dry-run).
type Sender struct {
	url, token string
	q          chan Event
	http       *http.Client
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
}

func NewSender(url, token string, buf int) *Sender {
	ctx, cancel := context.WithCancel(context.Background())
	s := &Sender{
		url: url, token: token, q: make(chan Event, buf),
		http: &http.Client{Timeout: 10 * time.Second}, ctx: ctx, cancel: cancel,
	}
	s.wg.Add(1)
	go s.run()
	return s
}

// Send не блокирует пайплайн: при переполненной очереди событие теряется.
func (s *Sender) Send(e Event) {
	select {
	case s.q <- e:
	default:
		log.Printf("sender: очередь переполнена, событие %s %s потеряно", e.Type, e.CourierID)
	}
}

// Close дожидается отправки очереди, но не дольше timeout.
func (s *Sender) Close(timeout time.Duration) {
	close(s.q)
	done := make(chan struct{})
	go func() { s.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		s.cancel()
		<-done
	}
}

func (s *Sender) run() {
	defer s.wg.Done()
	for e := range s.q {
		s.deliver(e)
	}
}

func (s *Sender) deliver(e Event) {
	body, _ := json.Marshal(e)
	if s.url == "" {
		e.CropJPEG = nil // в dry-run не засоряем вывод base64
		b, _ := json.Marshal(e)
		fmt.Println(string(b))
		return
	}
	backoff := time.Second
	for attempt := 1; ; attempt++ {
		err := s.post(body)
		if err == nil {
			log.Printf("sender: %s %s отправлено", e.Type, e.CourierID)
			return
		}
		if _, permanent := err.(permanentErr); permanent {
			log.Printf("sender: %s отброшено: %v", e.EventID, err)
			return
		}
		log.Printf("sender: попытка %d: %v, повтор через %s", attempt, err, backoff)
		select {
		case <-time.After(backoff):
		case <-s.ctx.Done():
			log.Printf("sender: остановка, событие %s не доставлено", e.EventID)
			return
		}
		if backoff < time.Minute {
			backoff *= 2
		}
	}
}

type permanentErr struct{ error }

func (s *Sender) post(body []byte) error {
	req, err := http.NewRequestWithContext(s.ctx, http.MethodPost, s.url, bytes.NewReader(body))
	if err != nil {
		return permanentErr{err}
	}
	req.Header.Set("Content-Type", "application/json")
	if s.token != "" {
		req.Header.Set("Authorization", "Bearer "+s.token)
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	switch {
	case resp.StatusCode < 300:
		return nil
	case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	default: // 4xx — повтор не поможет
		return permanentErr{fmt.Errorf("HTTP %d", resp.StatusCode)}
	}
}

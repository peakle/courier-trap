package main

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// Event — то, что уходит на бэкенд.
type Event struct {
	EventID   string    `json:"event_id"` // для идемпотентности на бэкенде
	Type      string    `json:"type"`     // visit_confirmed | visit_closed
	VisitID   string    `json:"visit_id"`
	CameraID  string    `json:"camera_id"`
	CourierID string    `json:"courier_id"`
	FirstSeen time.Time `json:"first_seen"`
	LastSeen  time.Time `json:"last_seen"`
	Reads     int       `json:"reads"`               // сколько кадров дали валидное чтение
	Agreement float64   `json:"agreement"`           // доля голосов за итоговые цифры, 0..1
	OCRConf   float64   `json:"ocr_conf"`            // лучшая уверенность OCR, 0..100
	CropJPEG  []byte    `json:"crop_jpeg,omitempty"` // base64 в JSON
}

// Visit — пребывание одного курьера в кадре.
type Visit struct {
	ID        string
	VisitID   string
	First     time.Time
	Last      time.Time
	Reads     int
	pos       [][10]int // голоса по позициям: pos[i][цифра]
	BestConf  float64
	BestCrop  []byte
	Confirmed bool
}

func (v *Visit) add(r Read, at time.Time) {
	v.Reads++
	v.Last = at
	for i := 0; i < len(r.ID) && i < len(v.pos); i++ {
		v.pos[i][r.ID[i]-'0']++
	}
	if r.Conf > v.BestConf {
		v.BestConf, v.BestCrop = r.Conf, r.Crop
	}
	// Итоговый ID — победитель голосования по каждой позиции.
	b := []byte(v.ID)
	for i := range v.pos {
		best := 0
		for d := 1; d < 10; d++ {
			if v.pos[i][d] > v.pos[i][best] {
				best = d
			}
		}
		b[i] = byte('0' + best)
	}
	v.ID = string(b)
}

// agreement — средняя по позициям доля голосов за победившую цифру.
func (v *Visit) agreement() float64 {
	sum := 0.0
	for i := range v.pos {
		max, tot := 0, 0
		for _, n := range v.pos[i] {
			tot += n
			if n > max {
				max = n
			}
		}
		if tot > 0 {
			sum += float64(max) / float64(tot)
		}
	}
	return sum / float64(len(v.pos))
}

// Tracker склеивает чтения в визиты и генерирует события.
type Tracker struct {
	CameraID   string
	MinReads   int           // чтений до подтверждения визита
	MinAgree   float64       // минимальное согласие голосов
	CloseAfter time.Duration // визит закрыт, если номер не виден столько времени
	Emit       func(Event)

	open []*Visit
}

func (t *Tracker) Add(r Read, at time.Time) {
	v := t.match(r.ID)
	if v == nil {
		v = &Visit{ID: r.ID, VisitID: randID(), First: at, pos: make([][10]int, len(r.ID))}
		t.open = append(t.open, v)
	}
	v.add(r, at)
	if !v.Confirmed && v.Reads >= t.MinReads && v.agreement() >= t.MinAgree {
		v.Confirmed = true
		t.Emit(t.event("visit_confirmed", v, true))
	}
}

// match: точное совпадение, иначе открытый визит с расстоянием Хэмминга 1
// (типичная ошибка OCR — одна цифра). Берём визит с наибольшим числом чтений.
func (t *Tracker) match(id string) *Visit {
	var best *Visit
	for _, v := range t.open {
		if v.ID == id {
			return v
		}
		if len(v.ID) == len(id) && hamming(v.ID, id) == 1 && (best == nil || v.Reads > best.Reads) {
			best = v
		}
	}
	return best
}

// Tick закрывает визиты, которые давно не видны. Вызывать регулярно.
func (t *Tracker) Tick(now time.Time) {
	keep := t.open[:0]
	for _, v := range t.open {
		if now.Sub(v.Last) > t.CloseAfter {
			if v.Confirmed {
				t.Emit(t.event("visit_closed", v, false))
			} // неподтверждённые визиты — шум, молча выбрасываем
			continue
		}
		keep = append(keep, v)
	}
	t.open = keep
}

// Flush закрывает все визиты (при остановке программы).
func (t *Tracker) Flush() {
	for _, v := range t.open {
		if v.Confirmed {
			t.Emit(t.event("visit_closed", v, false))
		}
	}
	t.open = nil
}

func (t *Tracker) event(typ string, v *Visit, withCrop bool) Event {
	e := Event{
		EventID: randID(), Type: typ, VisitID: v.VisitID, CameraID: t.CameraID,
		CourierID: v.ID, FirstSeen: v.First, LastSeen: v.Last,
		Reads: v.Reads, Agreement: v.agreement(), OCRConf: v.BestConf,
	}
	if withCrop {
		e.CropJPEG = v.BestCrop
	}
	return e
}

func hamming(a, b string) int {
	n := 0
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n
}

func randID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

package main

import (
	"fmt"
	"image"
	"image/color"
	"runtime"

	"gocv.io/x/gocv"
)

// На macOS окно OpenCV работает только из главного потока ОС.
func init() { runtime.LockOSThread() }

// Шрифт Hershey умеет только ASCII, поэтому подписи на латинице.
var (
	colorRead = color.RGBA{0, 220, 0, 0}     // номер прочитан
	colorCand = color.RGBA{255, 210, 0, 0}   // кандидат найден, но OCR отклонил
	colorROI  = color.RGBA{0, 200, 255, 0}   // зона обработки
	colorText = color.RGBA{255, 255, 255, 0} // статус визитов
)

const previewMaxW = 1280 // шире окно не делаем, кадр уменьшаем

// Preview — окно с разметкой. Все методы вызывать из главной горутины.
type Preview struct {
	win    *gocv.Window
	scaled gocv.Mat
}

func NewPreview(title string) *Preview {
	return &Preview{win: gocv.NewWindow(title), scaled: gocv.NewMat()}
}

func (p *Preview) Close() {
	p.scaled.Close()
	p.win.Close()
}

// Show рисует кадр в окне. false — пользователь нажал q/Esc или закрыл окно.
func (p *Preview) Show(frame gocv.Mat) bool {
	img := frame
	if frame.Cols() > previewMaxW {
		h := frame.Rows() * previewMaxW / frame.Cols()
		gocv.Resize(frame, &p.scaled, image.Pt(previewMaxW, h), 0, 0, gocv.InterpolationArea)
		img = p.scaled
	}
	if err := p.win.IMShow(img); err != nil {
		return false
	}
	switch p.win.WaitKey(1) {
	case 'q', 'Q', 27:
		return false
	}
	return p.win.IsOpen()
}

// annotate рисует разметку прямо в кадре. roi — зона обработки в координатах
// кадра; координаты чтений и кандидатов отсчитываются от её левого верхнего угла.
func annotate(frame gocv.Mat, roi image.Rectangle, reads []Read, rejected []Candidate, visits []VisitState) {
	if roi.Dx() < frame.Cols() || roi.Dy() < frame.Rows() {
		gocv.Rectangle(&frame, roi, colorROI, 1)
	}
	for _, c := range rejected {
		gocv.Rectangle(&frame, c.Rect.Add(roi.Min), colorCand, 2)
	}
	for _, r := range reads {
		rect := r.Rect.Add(roi.Min)
		gocv.Rectangle(&frame, rect, colorRead, 2)
		label(frame, fmt.Sprintf("%s %.0f%%", r.ID, r.Conf), image.Pt(rect.Min.X, rect.Min.Y-6), colorRead)
	}
	for i, v := range visits {
		state := "pending"
		if v.Confirmed {
			state = "CONFIRMED"
		}
		text := fmt.Sprintf("%s reads=%d agree=%.0f%% %s", v.ID, v.Reads, v.Agreement*100, state)
		label(frame, text, image.Pt(10, 24+i*24), colorText)
	}
}

// label — текст на тёмной подложке, чтобы читался на любом фоне.
func label(frame gocv.Mat, text string, at image.Point, c color.RGBA) {
	const scale, thickness = 0.6, 1
	size := gocv.GetTextSize(text, gocv.FontHersheySimplex, scale, thickness)
	bg := image.Rect(at.X-3, at.Y-size.Y-5, at.X+size.X+3, at.Y+5).
		Intersect(image.Rect(0, 0, frame.Cols(), frame.Rows()))
	if bg.Empty() {
		return
	}
	gocv.Rectangle(&frame, bg, color.RGBA{}, -1)
	gocv.PutText(&frame, text, at, gocv.FontHersheySimplex, scale, c, thickness)
}

package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"regexp"
	"strings"

	"github.com/otiai10/gosseract/v2"
	"gocv.io/x/gocv"
)

// Read — одно распознавание номера на одном кадре.
type Read struct {
	ID   string
	Conf float64 // уверенность Tesseract, 0..100
	Rect image.Rectangle
	Crop []byte // JPEG цветного кропа (для бэкенда)
}

// Reader — OCR строки цифр через Tesseract. НЕ потокобезопасен.
type Reader struct {
	client  *gosseract.Client
	re      *regexp.Regexp
	minConf float64
	targetH int  // к какой высоте цифры масштабируем кроп
	Verbose bool // логировать отклонённые кандидаты (-v)
}

func NewReader(pattern string, minConf float64) (*Reader, error) {
	c := gosseract.NewClient()
	if err := c.SetPageSegMode(gosseract.PSM_SINGLE_LINE); err != nil {
		c.Close()
		return nil, err
	}
	if err := c.SetWhitelist("0123456789"); err != nil {
		c.Close()
		return nil, err
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("плохой -pattern: %w", err)
	}
	return &Reader{client: c, re: re, minConf: minConf, targetH: 48}, nil
}

func (r *Reader) Close() error { return r.client.Close() }

func (r *Reader) logf(format string, a ...any) {
	if r.Verbose {
		log.Printf(format, a...)
	}
}

func (r *Reader) Read(bgr, gray gocv.Mat, c Candidate) (Read, bool) {
	frame := image.Rect(0, 0, gray.Cols(), gray.Rows())
	h := c.DigitH
	if h == 0 {
		h = c.Rect.Dy()
	}
	// По горизонтали поле шире: цепочка блобов часто теряет крайнюю цифру.
	padY := int(0.35*float64(h)) + 2
	padX := int(0.8*float64(h)) + 2
	rect := image.Rect(c.Rect.Min.X-padX, c.Rect.Min.Y-padY, c.Rect.Max.X+padX, c.Rect.Max.Y+padY).Intersect(frame)
	if rect.Empty() {
		return Read{}, false
	}

	roi := gray.Region(rect)
	work := roi.Clone()
	roi.Close()
	defer work.Close()

	// 1. Выравниваем наклон строки.
	if math.Abs(c.Angle) > 1.5 {
		m := gocv.GetRotationMatrix2D(image.Pt(work.Cols()/2, work.Rows()/2), c.Angle, 1)
		rot := gocv.NewMat()
		gocv.WarpAffineWithParams(work, &rot, m, image.Pt(work.Cols(), work.Rows()),
			gocv.InterpolationCubic, gocv.BorderReplicate, color.RGBA{})
		m.Close()
		work.Close()
		work = rot
	}

	// 2. Масштабируем к высоте цифры ~48 px — оптимум для Tesseract.
	s := float64(r.targetH) / float64(h)
	scaled := gocv.NewMat()
	defer scaled.Close()
	gocv.Resize(work, &scaled, image.Point{}, s, s, gocv.InterpolationCubic)

	// 3. Локальное выравнивание контраста. Без бинаризации: на ночных кадрах
	// Оцу съедает тонкие штрихи, а LSTM Tesseract нормально читает серое.
	eq := gocv.NewMat()
	defer eq.Close()
	clahe := gocv.NewCLAHEWithParams(2, image.Pt(8, 8))
	clahe.Apply(scaled, &eq)
	clahe.Close()
	if eq.Mean().Val1 < 110 {
		gocv.BitwiseNot(eq, &eq) // светлые цифры на тёмном фоне
	}
	padded := gocv.NewMat()
	defer padded.Close()
	gocv.CopyMakeBorder(eq, &padded, 20, 20, 20, 20, gocv.BorderConstant, color.RGBA{255, 255, 255, 0})

	png, err := gocv.IMEncode(gocv.PNGFileExt, padded)
	if err != nil {
		return Read{}, false
	}
	defer png.Close()
	if err := r.client.SetImageFromBytes(png.GetBytes()); err != nil {
		return Read{}, false
	}
	text, err := r.client.Text()
	if err != nil {
		return Read{}, false
	}
	id := strings.Join(strings.Fields(text), "")
	if !r.re.MatchString(id) {
		r.logf("  ocr %v: текст %q не подошёл под -pattern", c.Rect, id)
		return Read{}, false
	}
	// Уверенность строки Tesseract на таких кропах часто равна нулю,
	// поэтому усредняем по символам.
	conf := 0.0
	boxes, err := r.client.GetBoundingBoxes(gosseract.RIL_SYMBOL)
	if err != nil || len(boxes) == 0 {
		return Read{}, false
	}
	for _, b := range boxes {
		conf += b.Confidence
	}
	conf /= float64(len(boxes))
	if conf < r.minConf {
		r.logf("  ocr %v: %s уверенность %.0f < %.0f", c.Rect, id, conf, r.minConf)
		return Read{}, false
	}

	crop := bgr.Region(rect)
	defer crop.Close()
	jpg, err := gocv.IMEncode(gocv.JPEGFileExt, crop)
	var cropBytes []byte
	if err == nil {
		cropBytes = append([]byte(nil), jpg.GetBytes()...) // копия: буфер в C-памяти
		jpg.Close()
	}
	return Read{ID: id, Conf: conf, Rect: rect, Crop: cropBytes}, true
}

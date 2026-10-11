package main

import (
	"fmt"
	"image"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

// Candidate — область кадра, в которой предположительно находится номер.
type Candidate struct {
	Rect   image.Rectangle // bbox строки цифр в координатах кадра
	Angle  float64         // наклон строки в градусах (+ = вниз вправо)
	DigitH int             // высота цифры, px (0 — неизвестна)
}

// Detector ищет номера на кадре. bgr и gray — один и тот же кадр.
type Detector interface {
	Detect(bgr, gray gocv.Mat) []Candidate
	Close() error
}

// ---------------------------------------------------------------------------
// ChainDetector — эвристика без нейросети: ищет цепочку из 4–6 «цифроподобных»
// тёмных блобов одинаковой высоты, стоящих в строку. Хорошо работает на белой
// наклейке с тёмными цифрами (как у tezkor). Для продакшена лучше YOLO.
// ---------------------------------------------------------------------------

type ChainDetector struct {
	MinDigitH    int     // минимальная высота цифры, px
	MaxDigitFrac float64 // максимальная высота цифры как доля высоты кадра
	MinLen       int     // минимум блобов в цепочке
	MaxLen       int     // максимум блобов в цепочке
}

func (d *ChainDetector) Close() error { return nil }

func (d *ChainDetector) Detect(_, gray gocv.Mat) []Candidate {
	return d.detectIn(gray, image.Point{}, d.maxDigitH(gray))
}

// zoneDetector — детектор, умеющий искать номер только в заданных местах кадра
// (маска фиолетового короба).
type zoneDetector interface {
	DetectZones(gray, mask gocv.Mat) []Candidate
}

// DetectZones ищет только внутри зон короба — координаты кандидатов в системе
// кадра. Предел высоты цифры считается от всего кадра: внутри зоны цифра
// занимает заметную долю высоты, и доля от зоны отбросила бы её как слишком крупную.
func (d *ChainDetector) DetectZones(gray, mask gocv.Mat) []Candidate {
	maxH := d.maxDigitH(gray)
	var out []Candidate
	for _, z := range purpleZones(mask, d.MinDigitH) {
		sub := gray.Region(z)
		out = append(out, d.detectIn(sub, z.Min, maxH)...)
		sub.Close()
	}
	return out
}

func (d *ChainDetector) maxDigitH(gray gocv.Mat) int {
	return int(float64(gray.Rows()) * d.MaxDigitFrac)
}

// detectIn работает на gray (может быть ROI), off — смещение ROI в кадре.
func (d *ChainDetector) detectIn(gray gocv.Mat, off image.Point, maxH int) []Candidate {
	bin := gocv.NewMat()
	defer bin.Close()
	gocv.AdaptiveThreshold(gray, &bin, 255, gocv.AdaptiveThresholdMean, gocv.ThresholdBinaryInv, 31, 15)

	contours := gocv.FindContours(bin, gocv.RetrievalList, gocv.ChainApproxSimple)
	defer contours.Close()

	blobs := make([]image.Rectangle, 0, 256)
	for i := 0; i < contours.Size(); i++ {
		r := gocv.BoundingRect(contours.At(i))
		w, h := r.Dx(), r.Dy()
		if h < d.MinDigitH || h > maxH {
			continue
		}
		if ar := float64(w) / float64(h); ar < 0.2 || ar > 1.0 {
			continue
		}
		blobs = append(blobs, r)
	}
	sort.Slice(blobs, func(i, j int) bool { return blobs[i].Min.X < blobs[j].Min.X })

	var chains [][]image.Rectangle
	for i := range blobs {
		chain := []image.Rectangle{blobs[i]}
		cur := blobs[i]
		for j := i + 1; j < len(blobs) && len(chain) < d.MaxLen+1; j++ {
			n := blobs[j]
			h := float64(cur.Dy())
			gap := float64(n.Min.X - cur.Max.X)
			if gap > 2*h {
				break // дальше по X только дальше
			}
			if gap > 0.8*h || gap < -0.3*h { // слишком далеко или перекрывает текущую цифру
				continue
			}
			if math.Abs(float64(n.Dy())-h) > 0.25*h {
				continue
			}
			if math.Abs(cy(n)-cy(cur)) > 0.4*h {
				continue
			}
			chain = append(chain, n)
			cur = n
		}
		if len(chain) >= d.MinLen && len(chain) <= d.MaxLen {
			chains = append(chains, chain)
		}
	}

	// Длинные цепочки важнее; вложенные подцепочки отбрасываем.
	sort.Slice(chains, func(i, j int) bool {
		if len(chains[i]) != len(chains[j]) {
			return len(chains[i]) > len(chains[j])
		}
		return area(union(chains[i])) > area(union(chains[j]))
	})
	var out []Candidate
	var kept []image.Rectangle
	for _, ch := range chains {
		u := union(ch)
		dup := false
		for _, k := range kept {
			if overlapMin(u, k) > 0.5 {
				dup = true
				break
			}
		}
		if dup {
			continue
		}
		kept = append(kept, u)
		first, last := ch[0], ch[len(ch)-1]
		angle := math.Atan2(cy(last)-cy(first), cx(last)-cx(first)) * 180 / math.Pi
		hSum := 0
		for _, b := range ch {
			hSum += b.Dy()
		}
		out = append(out, Candidate{Rect: u.Add(off), Angle: angle, DigitH: hSum / len(ch)})
	}
	return out
}

// ---------------------------------------------------------------------------
// YOLODetector — YOLOv8/v11 в ONNX (один или несколько классов, выход [1,4+nc,N]).
// Найденный бокс уточняется ChainDetector'ом внутри ROI (строка цифр + наклон).
// ---------------------------------------------------------------------------

type YOLODetector struct {
	net    gocv.Net
	size   int
	conf   float32
	nms    float64
	refine *ChainDetector
}

func NewYOLODetector(path string, size int, conf float32) (*YOLODetector, error) {
	net := gocv.ReadNetFromONNX(path)
	if net.Empty() {
		return nil, fmt.Errorf("не удалось загрузить модель %s", path)
	}
	return &YOLODetector{
		net: net, size: size, conf: conf, nms: 0.45,
		refine: &ChainDetector{MinDigitH: 8, MaxDigitFrac: 1.0, MinLen: 4, MaxLen: 6},
	}, nil
}

func (d *YOLODetector) Close() error { return d.net.Close() }

func (d *YOLODetector) Detect(bgr, gray gocv.Mat) []Candidate {
	blob := gocv.BlobFromImage(bgr, 1.0/255, image.Pt(d.size, d.size), gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()
	d.net.SetInput(blob, "")
	out := d.net.Forward("")
	defer out.Close()

	dims := out.Size()
	if len(dims) != 3 || dims[1] < 5 {
		return nil
	}
	ch, n := dims[1], dims[2]
	data, err := out.DataPtrFloat32()
	if err != nil {
		return nil
	}
	sx := float32(bgr.Cols()) / float32(d.size)
	sy := float32(bgr.Rows()) / float32(d.size)
	frame := image.Rect(0, 0, bgr.Cols(), bgr.Rows())

	var rects []image.Rectangle
	var scores []float32
	for i := 0; i < n; i++ {
		best := float32(0)
		for c := 4; c < ch; c++ {
			if v := data[c*n+i]; v > best {
				best = v
			}
		}
		if best < d.conf {
			continue
		}
		cxv, cyv, w, h := data[i], data[n+i], data[2*n+i], data[3*n+i]
		r := image.Rect(int((cxv-w/2)*sx), int((cyv-h/2)*sy), int((cxv+w/2)*sx), int((cyv+h/2)*sy)).Intersect(frame)
		if r.Empty() {
			continue
		}
		rects = append(rects, r)
		scores = append(scores, best)
	}

	var res []Candidate
	for _, i := range nms(rects, scores, d.nms) {
		r := rects[i]
		roi := gray.Region(r)
		refined := d.refine.detectIn(roi, r.Min, roi.Rows())
		roi.Close()
		if len(refined) > 0 {
			res = append(res, refined[0])
		} else {
			res = append(res, Candidate{Rect: r})
		}
	}
	return res
}

// ---------------------------------------------------------------------------

func cx(r image.Rectangle) float64 { return float64(r.Min.X+r.Max.X) / 2 }
func cy(r image.Rectangle) float64 { return float64(r.Min.Y+r.Max.Y) / 2 }
func area(r image.Rectangle) int   { return r.Dx() * r.Dy() }

func union(rs []image.Rectangle) image.Rectangle {
	u := rs[0]
	for _, r := range rs[1:] {
		u = u.Union(r)
	}
	return u
}

// overlapMin — площадь пересечения / площадь меньшего прямоугольника.
func overlapMin(a, b image.Rectangle) float64 {
	in := area(a.Intersect(b))
	m := area(a)
	if area(b) < m {
		m = area(b)
	}
	if m == 0 {
		return 0
	}
	return float64(in) / float64(m)
}

func iou(a, b image.Rectangle) float64 {
	in := area(a.Intersect(b))
	un := area(a) + area(b) - in
	if un == 0 {
		return 0
	}
	return float64(in) / float64(un)
}

// nms — простое жадное подавление немаксимумов (без cgo).
func nms(rects []image.Rectangle, scores []float32, thr float64) []int {
	idx := make([]int, len(rects))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return scores[idx[a]] > scores[idx[b]] })
	var keep []int
	for _, i := range idx {
		ok := true
		for _, k := range keep {
			if iou(rects[i], rects[k]) > thr {
				ok = false
				break
			}
		}
		if ok {
			keep = append(keep, i)
		}
	}
	return keep
}

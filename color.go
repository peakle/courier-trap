package main

import (
	"image"

	"gocv.io/x/gocv"
)

// Номер всегда наклеен на фирменный короб, поэтому цепочки цифр ищем только
// в зонах его цвета: вывески и номера машин отсекаются, а кадр обрабатывается
// быстрее. Оттенок гуляет широко: синий короб нового дизайна H~112, сине-фиолетовый
// uzum bank H~119, пурпурный tezkor H~170. Нижний порог яркости отрезает тёмный
// фон (окна, небо), который иначе слипается с коробом в одно пятно.
const (
	purpleHMin = 105
	purpleHMax = 175
	purpleSMin = 45
	purpleVMin = 100
)

// purpleMask — бинарная маска фиолетового в кадре.
func purpleMask(bgr gocv.Mat) gocv.Mat {
	hsv := gocv.NewMat()
	defer hsv.Close()
	gocv.CvtColor(bgr, &hsv, gocv.ColorBGRToHSV)
	mask := gocv.NewMat()
	gocv.InRangeWithScalar(hsv,
		gocv.NewScalar(purpleHMin, purpleSMin, purpleVMin, 0),
		gocv.NewScalar(purpleHMax, 255, 255, 0), &mask)
	// Открытие убирает рябь (листва, блики, окна): короб — сплошное пятно.
	k := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer k.Close()
	gocv.MorphologyEx(mask, &mask, gocv.MorphOpen, k)
	return mask
}

// purpleZones — зоны поиска номера: прямоугольники фиолетовых пятен, расширенные
// на поля (наклейка бывает выше края короба) и объединённые при пересечении.
func purpleZones(mask gocv.Mat, minDigitH int) []image.Rectangle {
	cs := gocv.FindContours(mask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer cs.Close()
	frame := image.Rect(0, 0, mask.Cols(), mask.Rows())

	var zones []image.Rectangle
	for i := 0; i < cs.Size(); i++ {
		r := gocv.BoundingRect(cs.At(i))
		if r.Dy() < minDigitH || r.Dx() < 2*minDigitH {
			continue // пятно мельче строки цифр
		}
		pad := r.Dy() / 4
		if pad < 2*minDigitH {
			pad = 2 * minDigitH
		}
		zones = append(zones, r.Inset(-pad).Intersect(frame))
	}

	// Пересекающиеся зоны склеиваем, иначе один номер найдётся дважды.
	for merged := true; merged; {
		merged = false
		for i := 0; i < len(zones) && !merged; i++ {
			for j := i + 1; j < len(zones); j++ {
				if zones[i].Overlaps(zones[j]) {
					zones[i] = zones[i].Union(zones[j])
					zones = append(zones[:j], zones[j+1:]...)
					merged = true
					break
				}
			}
		}
	}
	return zones
}

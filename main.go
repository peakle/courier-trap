// courier-trap: распознаёт 5-значный номер курьера на термокоробе
// и отправляет события визитов на бэкенд.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"gocv.io/x/gocv"
)

type config struct {
	src, images, cameraID, backend, token, model, roi, pattern string
	fps, motion, minConf, minAgree                             float64
	idle, closeAfter                                           time.Duration
	minDigitH, minReads, modelSize                             int
	modelConf                                                  float64
	verbose                                                    bool
}

func main() {
	var c config
	flag.StringVar(&c.src, "src", "", "видеопоток: rtsp://..., файл или индекс устройства")
	flag.StringVar(&c.images, "image", "", "тест: распознать фото (через запятую) и выйти")
	flag.StringVar(&c.cameraID, "camera-id", "cam-1", "идентификатор точки")
	flag.StringVar(&c.backend, "backend", "", "URL для POST событий (пусто — печать в stdout)")
	flag.StringVar(&c.token, "token", os.Getenv("BACKEND_TOKEN"), "Bearer-токен (или env BACKEND_TOKEN)")
	flag.StringVar(&c.model, "model", "", "YOLO ONNX детектор номера (пусто — эвристика)")
	flag.IntVar(&c.modelSize, "model-size", 640, "входной размер модели")
	flag.Float64Var(&c.modelConf, "model-conf", 0.4, "порог уверенности детектора")
	flag.StringVar(&c.roi, "roi", "", "зона обработки x,y,w,h (пусто — весь кадр)")
	flag.StringVar(&c.pattern, "pattern", `^\d{5}$`, "регулярка допустимого номера")
	flag.Float64Var(&c.fps, "fps", 2, "сколько кадров в секунду анализировать")
	flag.Float64Var(&c.motion, "motion", 0.002, "доля изменившихся пикселей для запуска распознавания")
	flag.DurationVar(&c.idle, "idle", 5*time.Second, "распознавать и без движения раз в N (стоящие курьеры)")
	flag.DurationVar(&c.closeAfter, "close-after", 3*time.Minute, "визит закрыт, если номер не виден N")
	flag.IntVar(&c.minDigitH, "min-digit-h", 12, "минимальная высота цифры, px")
	flag.IntVar(&c.minReads, "min-reads", 5, "чтений до подтверждения визита")
	flag.Float64Var(&c.minAgree, "min-agree", 0.7, "минимальное согласие голосов 0..1")
	flag.Float64Var(&c.minConf, "min-conf", 50, "минимальная уверенность OCR 0..100")
	flag.BoolVar(&c.verbose, "v", false, "логировать каждое чтение")
	flag.Parse()

	det, err := newDetector(c)
	if err != nil {
		log.Fatal(err)
	}
	defer det.Close()
	reader, err := NewReader(c.pattern, c.minConf)
	if err != nil {
		log.Fatal(err)
	}
	defer reader.Close()

	switch {
	case c.images != "":
		os.Exit(runImages(c, det, reader))
	case c.src != "":
		runStream(c, det, reader)
	default:
		flag.Usage()
		os.Exit(2)
	}
}

func newDetector(c config) (Detector, error) {
	if c.model != "" {
		return NewYOLODetector(c.model, c.modelSize, float32(c.modelConf))
	}
	return &ChainDetector{MinDigitH: c.minDigitH, MaxDigitFrac: 0.08, MinLen: 4, MaxLen: 6}, nil
}

// recognize — один кадр: детекция + OCR. Координаты — относительно frame.
func recognize(frame gocv.Mat, det Detector, reader *Reader) []Read {
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(frame, &gray, gocv.ColorBGRToGray)
	var reads []Read
	for _, cand := range det.Detect(frame, gray) {
		if r, ok := reader.Read(frame, gray, cand); ok {
			reads = append(reads, r)
		}
	}
	return reads
}

// ---------------------------------------------------------------- фото-режим

func runImages(c config, det Detector, reader *Reader) int {
	found := 0
	for _, path := range strings.Split(c.images, ",") {
		img := gocv.IMRead(strings.TrimSpace(path), gocv.IMReadColor)
		if img.Empty() {
			log.Printf("%s: не удалось прочитать", path)
			continue
		}
		frame, closeROI := applyROI(img, c.roi)
		start := time.Now()
		reads := recognize(frame, det, reader)
		elapsed := time.Since(start)
		closeROI()
		img.Close()
		for _, r := range reads {
			found++
			out, _ := json.Marshal(map[string]any{
				"file": path, "courier_id": r.ID, "ocr_conf": r.Conf,
				"rect": []int{r.Rect.Min.X, r.Rect.Min.Y, r.Rect.Dx(), r.Rect.Dy()},
				"ms":   elapsed.Milliseconds(),
			})
			fmt.Println(string(out))
		}
		if len(reads) == 0 {
			log.Printf("%s: номер не найден (%d ms)", path, elapsed.Milliseconds())
		}
	}
	if found == 0 {
		return 1
	}
	return 0
}

// ---------------------------------------------------------------- поток

func runStream(c config, det Detector, reader *Reader) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	sender := NewSender(c.backend, c.token, 1000)
	tracker := &Tracker{
		CameraID: c.cameraID, MinReads: c.minReads, MinAgree: c.minAgree,
		CloseAfter: c.closeAfter, Emit: sender.Send,
	}
	isFile := isFileSource(c.src)
	retry := time.Second

	for ctx.Err() == nil {
		if ended := processSource(ctx, c, isFile, det, reader, tracker); ended {
			break // файл закончился
		}
		if ctx.Err() == nil {
			log.Printf("поток недоступен, переподключение через %s", retry)
			select {
			case <-time.After(retry):
			case <-ctx.Done():
			}
			if retry < 30*time.Second {
				retry *= 2
			}
		}
	}
	tracker.Flush()
	sender.Close(10 * time.Second)
}

// processSource читает поток до ошибки. ended=true — штатный конец файла.
func processSource(ctx context.Context, c config, isFile bool, det Detector, reader *Reader, tr *Tracker) (ended bool) {
	vc, err := openCapture(c.src)
	if err != nil {
		log.Printf("open %s: %v", c.src, err)
		return false
	}
	defer vc.Close()
	log.Printf("источник открыт: %s", c.src)

	frame := gocv.NewMat()
	defer frame.Close()
	prev := gocv.NewMat()
	defer prev.Close()

	interval := time.Duration(float64(time.Second) / c.fps)
	start := time.Now()
	var nextTick, lastFull time.Time

	for ctx.Err() == nil {
		// Метка времени: для файла — время внутри видео, для потока — часы.
		now := time.Now()
		if isFile {
			now = start.Add(time.Duration(vc.Get(gocv.VideoCapturePosMsec) * float64(time.Millisecond)))
		}
		if now.Before(nextTick) {
			vc.Grab(1) // пропускаем кадр без конвертации, чтобы не копить задержку RTSP
			if isFile && vc.Get(gocv.VideoCapturePosFrames) >= vc.Get(gocv.VideoCaptureFrameCount) {
				return true
			}
			continue
		}
		if ok := vc.Read(&frame); !ok || frame.Empty() {
			return isFile
		}
		nextTick = now.Add(interval)

		work, closeROI := applyROI(frame, c.roi)
		moved := motionRatio(work, &prev) > c.motion
		if moved || now.Sub(lastFull) >= c.idle {
			lastFull = now
			reads := recognize(work, det, reader)
			if c.verbose {
				log.Printf("t=%s motion=%v reads=%d", now.Sub(start).Round(10*time.Millisecond), moved, len(reads))
			}
			for _, r := range reads {
				if c.verbose {
					log.Printf("  read %s conf=%.0f", r.ID, r.Conf)
				}
				tr.Add(r, now)
			}
		}
		closeROI()
		tr.Tick(now)
	}
	return false
}

// motionRatio — доля изменившихся пикселей относительно прошлого кадра (на 320 px).
func motionRatio(frame gocv.Mat, prev *gocv.Mat) float64 {
	small := gocv.NewMat()
	defer small.Close()
	w := 320
	h := frame.Rows() * w / frame.Cols()
	gocv.Resize(frame, &small, image.Pt(w, h), 0, 0, gocv.InterpolationArea)
	gocv.CvtColor(small, &small, gocv.ColorBGRToGray)
	gocv.GaussianBlur(small, &small, image.Pt(5, 5), 0, 0, gocv.BorderDefault)
	defer small.CopyTo(prev)
	if prev.Empty() || prev.Rows() != small.Rows() {
		return 1
	}
	diff := gocv.NewMat()
	defer diff.Close()
	gocv.AbsDiff(small, *prev, &diff)
	gocv.Threshold(diff, &diff, 25, 255, gocv.ThresholdBinary)
	return float64(gocv.CountNonZero(diff)) / float64(w*h)
}

func applyROI(frame gocv.Mat, roi string) (gocv.Mat, func()) {
	if roi == "" {
		return frame, func() {}
	}
	p := strings.Split(roi, ",")
	if len(p) != 4 {
		log.Fatalf("-roi: ожидается x,y,w,h")
	}
	v := make([]int, 4)
	for i := range p {
		n, err := strconv.Atoi(strings.TrimSpace(p[i]))
		if err != nil {
			log.Fatalf("-roi: %v", err)
		}
		v[i] = n
	}
	r := image.Rect(v[0], v[1], v[0]+v[2], v[1]+v[3]).Intersect(image.Rect(0, 0, frame.Cols(), frame.Rows()))
	if r.Empty() {
		log.Fatalf("-roi вне кадра")
	}
	sub := frame.Region(r)
	return sub, func() { sub.Close() }
}

func openCapture(src string) (*gocv.VideoCapture, error) {
	if n, err := strconv.Atoi(src); err == nil {
		return gocv.OpenVideoCapture(n)
	}
	return gocv.OpenVideoCaptureWithAPI(src, gocv.VideoCaptureFFmpeg)
}

func isFileSource(src string) bool {
	if _, err := strconv.Atoi(src); err == nil {
		return false
	}
	if strings.Contains(src, "://") {
		return false
	}
	_, err := os.Stat(src)
	return err == nil
}

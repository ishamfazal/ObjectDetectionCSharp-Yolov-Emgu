using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using ObjectDetection_C_.Abstracts;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectDetection_C_.Implemention
{
    public class ObjectDetectionRepository : Detections
    {
        public override void InitiateCascadeEmgu()
        {
            var facecasede = DnnInvoke.ReadNetFromDarknet("./YolovModels/yolov7-tiny.cfg", "./YolovModels/yolov7-tiny.weights");
            IList<string> classLabels = File.ReadLines("./YolovModels/coco.names").ToList();

            facecasede.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
            facecasede.SetPreferableTarget(Target.Cpu);

            var vc = new VideoCapture(0, VideoCapture.API.DShow);

            Mat frame = new();
            VectorOfMat output = new();
            VectorOfRect boxes = new();
            VectorOfFloat scores = new();
            VectorOfInt indices = new();

            while (true)
            {
                vc.Read(frame);

                CvInvoke.Resize(frame, frame, new System.Drawing.Size(0, 0), .4, .4);

                boxes = new();
                indices = new();
                scores = new();

                var image = frame.ToImage<Bgr, byte>();
                var input = DnnInvoke.BlobFromImage(image, (1 / 255.0), swapRB: true);

                facecasede.SetInput(input);
                facecasede.Forward(output, facecasede.UnconnectedOutLayersNames);

                for (int i = 0; i < output.Size; i++)
                {
                    var mat = output[i];
                    var data = (float[,])mat.GetData();

                    for (int j = 0; j < data.GetLength(0); j++)
                    {
                        float[] row = Enumerable.Range(0, data.GetLength(1)).Select(x => data[j, x]).ToArray();
                        var rowScore = row.Skip(5).ToArray();
                        var classId = rowScore.ToList().IndexOf(rowScore.Max());
                        var confidence = rowScore[classId];

                        if (confidence > 0.8f)
                        {
                            var centerX = (int)(row[0] * frame.Width);
                            var centerY = (int)(row[1] * frame.Height);
                            var boxWidth = (int)(row[2] * frame.Width);
                            var boxHeight = (int)(row[3] * frame.Height);

                            var x = (int)(centerX - (boxWidth / 2));
                            var y = (int)(centerY - (boxHeight / 2));

                            boxes.Push(new System.Drawing.Rectangle[] { new System.Drawing.Rectangle(x, y, boxWidth, boxHeight) });

                            indices.Push(new int[] { classId });
                            scores.Push(new float[] { confidence });
                        }

                    }
                }

                var bestIndex = DnnInvoke.NMSBoxes(boxes.ToArray(), scores.ToArray(), .8f, .8f);
                var frameout = frame.ToImage<Bgr, byte>();

                for (int i = 0; i < bestIndex.Length; i++)
                {
                    int index = bestIndex[i];
                    var box = boxes[index];
                    var label = classLabels[indices[index]];
                    CvInvoke.Rectangle(frameout, box, new MCvScalar(0, 255, 0), 2);
                    CvInvoke.PutText(frameout, label, new System.Drawing.Point(box.X, box.Y - 20), Emgu.CV.CvEnum.FontFace.HersheyComplexSmall, 1.0, new MCvScalar(0, 0, 255), 2);
                }

                CvInvoke.Resize(frameout, frameout, new System.Drawing.Size(0, 0), 4, 4);
                CvInvoke.Imshow("detections", frameout);

                if (CvInvoke.WaitKey(1) == 27)
                {
                    break;
                }
            }
        }
    }
}

using Emgu.CV.Structure;
using Emgu.CV;
using ObjectDetection_C_.Abstracts;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectDetection_C_.Implemention
{
    public class FaceDetectionRepository : Detections
    {
        public override void InitiateCascadeEmgu()
        {
            var facecasede = new CascadeClassifier("./YolovModels/haarcascade_frontalface_default.xml");

            var vc = new VideoCapture(0, VideoCapture.API.DShow);

            Mat frame = new();
            Mat framGray = new();

            while (true)
            {
                vc.Read(frame);

                CvInvoke.CvtColor(frame, framGray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                var faces = facecasede.DetectMultiScale(framGray, 1.3, 5);

                if (faces is not null && faces.Length > 0)
                {
                    CvInvoke.Rectangle(frame, faces[0], new MCvScalar(0, 255, 0), 2);
                }

                CvInvoke.Imshow("face detection", frame);

                if (CvInvoke.WaitKey(1) == 27)
                {
                    break;
                }
            }
        }
    }
}

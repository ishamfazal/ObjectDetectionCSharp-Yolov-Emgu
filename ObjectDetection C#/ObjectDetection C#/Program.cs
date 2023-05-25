using ObjectDetection_C_.Abstracts;
using ObjectDetection_C_.Implemention;
using System;

namespace ObjectDetection_C_
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Detections detections = new ObjectDetectionRepository();
            detections.InitiateCascadeEmgu();
        }
    }
}

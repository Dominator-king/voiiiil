import React, { useRef } from "react";
import { Client } from "@gradio/client";

const VideoStream = () => {
  const videoRef = useRef(null);

  const startVideo = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    videoRef.current.srcObject = stream;

    const peerConnection = new RTCPeerConnection();
    stream
      .getTracks()
      .forEach((track) => peerConnection.addTrack(track, stream));

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    console.log(peerConnection.localDescription);

    const client = await Client.connect("http://127.0.0.1:7860/");
    const result = await client.predict("/predict", {
      sdp: { foo: "bar" },
    });

    console.log(result.data);

    await peerConnection.setRemoteDescription(
      new RTCSessionDescription(result.data)
    );

    peerConnection.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
      }
    };
  };

  return (
    <div>
      <button onClick={startVideo}>Start Video</button>
      <video ref={videoRef} autoPlay playsInline />
    </div>
  );
};

export default VideoStream;

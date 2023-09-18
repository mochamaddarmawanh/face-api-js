async function start_face_api_js() {
    const img_pater_parker = document.getElementById('img_pater_parker')
    const img_input = document.getElementById('img_input')

    /* Loading the Models */
    await faceapi.nets.ssdMobilenetv1.loadFromUri('/face-api.js/weights')
    // await faceapi.nets.ageGenderNet.loadFromUri('/face-api.js/weights')
    // await faceapi.nets.faceExpressionNet.loadFromUri('/face-api.js/weights')
    await faceapi.nets.faceLandmark68Net.loadFromUri('/face-api.js/weights')
    await faceapi.nets.faceRecognitionNet.loadFromUri('/face-api.js/weights')

    /* Detecting Faces */
    // const detections = await faceapi.detectAllFaces(img_pater_parker).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptors()
    const detections = await faceapi.detectAllFaces(img_pater_parker).withFaceLandmarks().withFaceDescriptors()
    const labeledDescriptors = [
        new faceapi.LabeledFaceDescriptors(
            'Pater Parker',
            [detections[0].descriptor]
        )
    ]
    if (!detections.length) {
        return
    }

    /* Displaying Detection Results */
    const displaySize = { width: img_input.width, height: img_input.height }
    // resize the canvas canvas to the input dimensions
    const canvas = document.getElementById('canvas')
    faceapi.matchDimensions(canvas, displaySize)

    /* Draw the image onto canvas */
    canvas.getContext('2d').drawImage(img_input, 0, 0, displaySize.width, displaySize.height)

    /* Face Recognition by Matching Descriptors */
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors)
    const results = await faceapi.detectAllFaces(img_input).withFaceLandmarks().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(results, displaySize)
    results.forEach(fd => {
        const bestMatch = faceMatcher.findBestMatch(fd.descriptor);
        // new faceapi.draw.DrawTextField(bestMatch.label + ' (' + Math.round((1 - bestMatch.distance) * 100) + "%)", fd.detection.box.bottomLeft).draw(canvas);
        // console.log(bestMatch.toString());
        const accurasy = bestMatch.label === 'unknown' ? 0 : Math.round((1 - bestMatch.distance) * 100)
        new faceapi.draw.DrawBox(fd.detection.box, {
            label: bestMatch.label + ' (' + accurasy + '%)'
        }).draw(canvas)
    });

    // /* Draw Detection Results */
    // // resize the detected boxes in case your displayed image has a different size than the original
    // const resizedDetections = faceapi.resizeResults(results, displaySize)
    // // draw detections into the canvas
    // faceapi.draw.drawDetections(canvas, resizedDetections)
    // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
    // // faceapi.draw.drawFaceExpressions(canvas, resizedDetections, minProbability = 0.05)
}

start_face_api_js()
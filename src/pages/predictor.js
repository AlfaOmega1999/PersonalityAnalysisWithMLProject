
import React, { useEffect, useState } from 'react';
import FastAPIClient from '../client.js';
import config from '../config.js';
import { PushSpinner} from "react-spinners-kit";
import Swal from 'sweetalert2'
import "../style/predict.css"

const client = new FastAPIClient(config);
const available_types = ['ISTJ','ISFJ','INFJ','INTJ','ISTP','ISFP','INFP','INTP','ESTP','ESFP','ENFP','ENTP','ESTJ','ESFJ','ENFJ','ENTJ'];


const Predictor = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false)

  useEffect(() => {
  }, [])

  const Predict = (msg) => {
    animation1();
    setLoading(true);
    if(msg === null) msg = "Test message"
    client.getPrediction(msg).then((data) => {
      setLoading(false)
      result_alert(data);
      client.getTypeStats(data.prediction).then((data) => {
        localStorage.setItem('type_stats', JSON.stringify(data.type_stats))
        })
      })

  } 
  const result_alert = (data) =>{
    localStorage.setItem('prediction', JSON.stringify(data.prediction))
    Swal.fire({
      title: "You are a " + data.prediction +" type",
      imageWidth: 400,
      imageHeight: 200,
      showConfirmButton: true,
      confirmButtonText: 'Continue',
      showCloseButton: false,
      //backdrop: "linear-gradient(#0a39e0, #256ce1)",
      background: "white",
      })
      .then((result) => {
        Swal.fire({
          text: "Do you agree with your result?",
          imageWidth: 400,
          imageHeight: 200,
          showCancelButton: true,
          cancelButtonText: 'Change it',
          showConfirmButton: true,
          confirmButtonText: 'Yes',
          showCloseButton: false,
          icon: 'question',
          background: "white",
          }).then((result) => {
            if (!result.isConfirmed) {
              Swal.fire({
                title: 'Which type do you think you are?',
                text: 'By changing it you are allowing us to save your result, and helping us to keep improving our app',
                icon: 'question',
                html: `<input type="text" id="type" class="swal2-input" placeholder="New type">`,
                preConfirm: () => {
                  const new_type = Swal.getPopup().querySelector('#type').value
                  if (!new_type) {
                    Swal.showValidationMessage(`Please enter a type`)
                  }
                  if (!available_types.includes(new_type)) {
                    Swal.showValidationMessage(`Please enter a valid type`)
                  }
                  return { new_type: new_type}
                }
              }).then((result) => {
                client.updateDataset(result.value.new_type,data.newPost).then(() => {
                  Swal.fire("Successfully changed", "Your type suggestion has been saved", "success");
                  })
              });
            } 
            else {
              Swal.fire({
                text: "Do you want to help us by saving your result as new data?",
                imageWidth: 400,
                imageHeight: 200,
                showCancelButton: true,
                cancelButtonText: 'No, thanks',
                showConfirmButton: true,
                confirmButtonText: 'Yes, save it',
                showCloseButton: false,
                icon: 'question',
                background: "white",
                }).then((result) => {
                  if (result.isConfirmed) {
                    client.updateDataset(data.prediction,data.newPost).then(() => {
                      Swal.fire("Great!", "Your type suggestion has been saved, and it will be useful for improving our results", "info");
                    })
                  }
                });
            }
          })
      })
      
  };
  const animation1 = () =>{
    var toastMixin = Swal.mixin({
      toast: true,
      icon: 'success',
      title: 'General Title',
      animation: false,
      position: 'top-right',
      showConfirmButton: false,
      timer: 2000,
      timerProgressBar: true,
    });
      toastMixin.fire({
        animation: true,
        title: 'Your prediction has been sent'
      });
    };
function getData(val){
  setData(val.target.value)
}
if(loading){ 
  return(
    <header2 >
          <div>
            <PushSpinner   size={100} color= '#BE9533'  loading={loading} />
          </div>
    </header2>
  )
}
else{
  return (
    <header id="home">
      <div>
        <br /><br /><br /><br /><br /><br />
        <textarea placeholder="Write your message inside this box..." id="text" name="text" onChange={getData} rows="8"></textarea>  
        <br />
        <input id="button" className="first" type="submit" value="Predict" onClick={() => Predict(data)}></input>
      </div>
    </header>
  );
};
}

export default Predictor;
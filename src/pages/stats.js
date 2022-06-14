import React, { useEffect, useState } from 'react';
import FastAPIClient from '../client.js';
import config from '../config.js';
import 'chart.js/auto';
import {Pie} from 'react-chartjs-2';
import "../style/stats.css"

const client = new FastAPIClient(config);

const Stats = () => {
  const [staditics, setStadistics] = useState(null);

  const allStats = async() => {
    client.getAllStats().then((data) => {
      localStorage.setItem('all_stats', JSON.stringify(data.all_stats));
      setStadistics(JSON.parse(localStorage.getItem('all_stats')));
    })
  }
  const Statsdata ={
    labels:['ISTJ','ISFJ','INFJ','INTJ','ISTP','ISFP','INFP','INTP','ESTP','ESFP','ENFP','ENTP','ESTJ','ESFJ','ENFJ','ENTJ'],
    datasets:[{
      data: staditics,
      backgroundColor: ['#C0392B','#9B59B6','#839192','#17A589','#229954','#D4AC0D','#CA6F1E','#D0D3D4','#2471A3','#239B56','#7D3C98','#2E86C1','#2E4053','#BA4A00','#E74C3C','#16A085']
    }]
  };
  const opciones ={
    maintainAspectRatio: false,
    responsive:true,
    offset:true,
    plugins: {  // 'legend' now within object 'plugins {}'
      legend: {
        position: 'right',
        labels: {
          color: "white",
          font: {
            size: 15
          }
        }
      }
    },
  }

  useEffect(() => {
    allStats()
  }, [])
  
  if(staditics!= null){
    if(JSON.parse(localStorage.getItem('prediction')) != null){
      return (
        <header id="home">
          <div className="conatiner">
            <div className="wrap">
              <div className="statsbox">
                <div className="front">
                <h1 className="stats-type">{JSON.parse(localStorage.getItem('prediction'))}</h1>
                <h1 className="stats-type-text" >You are part of the</h1>
                <h1 className="stats-type-number" color='red'>{JSON.parse(localStorage.getItem('type_stats'))}%</h1>
                <h1 className="stats-type-text" >of the population</h1>
                </div>
              </div>
            </div>
            <div class="wrap">
              <div class="statsbox2" >
                  <h1 className="stats-text">General stats:</h1>
                    <div className='chart'>
                      <Pie data={Statsdata} options={opciones}/>
                    </div>  
              </div>
            </div>
          </div>
        </header>
      );
    }
    else{
      return (
        <header id="home">
          <div className="conatiner">
            <div class="wrap">
              <div class="statsbox2" >
                  <h1 className="stats-text">General stats:</h1>
                    <div className='chart'>
                      <Pie data={Statsdata} options={opciones}/>
                    </div>  
              </div>
            </div>
          </div>
        </header>
      );
    }
  };
}
  

export default Stats;
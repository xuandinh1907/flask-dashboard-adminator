window.addEventListener('load', function () {

    var submit_btn=this.document.getElementsByClassName("qa_submit")[0]
    submit_btn.addEventListener('click', function (event) {
        event.preventDefault();
        
        var paragraph_val = document.getElementById("input_paragraph").value;
        var question_val = document.getElementById("input_question").value;
        var answer_val = document.getElementById("input_answer").value;
        console.log (paragraph_val);
        console.log (question_val);
        console.log (answer_val);
        var asking_obj ={"para":paragraph_val, "ques":question_val}
        
        
        url = 'http://127.0.0.1:5000/qa_processing';
        // fetch(url)
        // .then(response => response.json())
        // .then(data => {console.log(data);
        //     display_info (data);
        // })

        fetch('http://127.0.0.1:5000/qa_processing', {
            method: 'post',
            headers: {
                'Content-Type': 'application/json'
              },
            body: JSON.stringify(asking_obj)
        }).then(function(response) {
            return response.json();
        }).then(function(data) {
            console.log(data);
            answer_text= "";
            var keys = Object.keys(data);
            for(var i=0; i<keys.length; i++){
                var key = keys[i];
                answer_text += "Question: "+key +"\nAnswer: "+data[key]+"\n";
                console.log(key, data[key]);
            }
            document.getElementById("input_answer").value = answer_text;

        });
    });
});


// window.addEventListener('load', function () {
//     getLocation_display();
// })

// function getLocation_display() {
//     if (navigator.geolocation) {
//     navigator.geolocation.getCurrentPosition(weather_ajax);
//     } else {
//     x.innerHTML = "Geolocation is not supported by this browser.";
//     }
// }

// function weather_ajax(position) {
//     const lat_long = {
//         "lat": position.coords.latitude ,
//         "long": position.coords.longitude
//     };
//     // console.log(lat_long)
//     ajax_get_weather(lat_long);
// }

// function ajax_get_weather(lat_long){
//     // console.log(lat_long)
//     url = 'http://127.0.0.1:5000/get_data_api.html?lat='+lat_long.lat +'&lon='+lat_long.long
//     console.log("Calling Ajax at "+url)
//     fetch(url)
//     .then(response => response.json())
//     .then(data => {console.log(data);
//         display_info (data);
//     })
// }

// function convert_time(t){
//     var today = new Date(t*1000);
//     var daylist = ["SUN","MON","TUE","WED","THU","FRI","SAT"];
//     var day = today.getDay();
//     var long_date = today.getDate()+'-'+(today.getMonth()+1)+'-'+today.getFullYear();
//     var short_date = today.getDate()+'-'+(today.getMonth()+1);
//     return {'day': daylist[day], 'long_date': long_date, 'short_date': short_date}
// }

// function F2C(fd){
//     let cd = (fd - 32) * (5/9);
//     return Math.round((fd - 32) * (5/9));
// }

// function display_info(weather_obj){
 
//     const timezone = weather_obj.timezone;
//     const currently = weather_obj.currently;
//     const daily = weather_obj.daily;
//     const hourly = weather_obj.hourly; 

//     // console.log(weather_obj)
//     date_obj = convert_time(currently.time)
//     today_day =date_obj.day;
//     today_date =date_obj.long_date;

//     let rain_chance_dom = document.querySelector(".top_panel .rain_chance");
//     let ozone_dom = document.querySelector(".top_panel .ozone");
//     let wind_direction_dom = document.querySelector(".top_panel .wind_direction");
//     let humidity_dom = document.querySelector('.top_panel .humidity'); 

//     rain_chance_dom.innerHTML = currently.precipProbability+'%';
//     ozone_dom.innerHTML = currently.ozone+'DU';
//     wind_direction_dom.innerHTML = currently.windBearing+'°';
//     humidity_dom.innerHTML = currently.humidity+'%';

//     let timezone_dom = document.querySelector(".weather-dashboard .timezone");
//     let today_temp_dom = document.querySelector(".weather-dashboard .today .temp");
//     let today_description_dom = document.querySelector(".weather-dashboard .today .description");
//     let today_day_dom = document.querySelector('.weather-dashboard .today .day');
//     let today_date_dom = document.querySelector('.weather-dashboard .today .date');

//     let today_windspeed_dom = document.querySelector('.weather-dashboard .extend .windspeed');
//     let today_uv_dom = document.querySelector('.weather-dashboard .extend .uv');
//     let today_pressure_dom = document.querySelector('.weather-dashboard .extend .pressure');
    
//     let weekly_dom = document.querySelector('.weather-dashboard .weekly');
//     let today_weather_icon_dom = document.getElementById('today-icon');
    
//     timezone_dom.innerHTML = timezone;
//     today_temp_dom.innerHTML = F2C(currently.apparentTemperature)+'<sup>°C</sup>';
//     today_description_dom.innerHTML = currently.summary;
//     today_day_dom.innerHTML = today_day;
//     today_date_dom.innerHTML = today_date;
//     today_windspeed_dom.innerHTML = currently.windSpeed;
//     today_uv_dom.innerHTML =  currently.uvIndex;
//     today_pressure_dom.innerHTML = currently.pressure;

//     var icons = new Skycons({"color": "orange"});

//     icons.set(today_weather_icon_dom, currently.icon);

//     weekly_dom.innerHTML = "";

//     i = 0;
//     let temp_week = [];
//     let day_week =[];
//     let max_temp = [];
//     let min_temp = []
//     for (const day_data of daily.data){

//         let temp = F2C((day_data.temperatureMin+day_data.temperatureMax)/2);
//         temp_week.push(temp);

//         daily_date = convert_time(day_data.time)
//         let day = daily_date.short_date;
//         day_week.push(day);

//         max_temp.push(F2C(day_data.temperatureMax));
//         min_temp.push(F2C(day_data.temperatureMin));

//         dom = '<div class="peer"><h6 class="mB-10">'+ day +'</h6><canvas id="'+ 
//         day_data.icon + i
//         +'" width="30" height="30"></canvas><span class="d-b fw-600">'+ 
//         temp
//         +'<sup>°C</sup></span></div>';

//         weekly_dom.insertAdjacentHTML( 'beforeend', dom);
//         // console.log(dom);
//         icons.set(document.getElementById(day_data.icon + i), day_data.icon);
//         i = i +1;
//     }
//     icons.play();

//     // console.log(temp_week);
//     // console.log(day_week);

//     var t = document.getElementById("line-chart-weather");
//     if (t) {
//         var i = t.getContext("2d");
//         t.height = 80, new Chart(i, {
//             type: "line",
//             data: {
//                 labels: day_week,
//                 datasets: [{
//                     label: "Max Temperature",
//                     backgroundColor: "rgba(237, 231, 246, 0.5)",
//                     borderColor: "#D56161",
//                     pointBackgroundColor: "#D43535",
//                     borderWidth: 2,
//                     data: max_temp
//                 }, {
//                     label: "Min Temperature",
//                     backgroundColor: "rgba(232, 245, 233, 0.5)",
//                     borderColor: "#2196f3",
//                     pointBackgroundColor: "#1976d2",
//                     borderWidth: 2,
//                     data: min_temp
//                 }]
//             },
//             options: {
//                 legend: {
//                     display: !1
//                 }
//             }
//         })
//     }
// }
// window.addEventListener('load', function () {

//     var submit_btn= document.getElementsByClassName("qa_submit")[0];

//     if (submit_btn != undefined){
//         submit_btn.addEventListener('click', function (event) {
//             event.preventDefault();
            
//             var paragraph_val = document.getElementById("input_paragraph").value;
//             var question_val = document.getElementById("input_question").value;
//             var answer_val = document.getElementById("input_answer").value;
//             console.log (paragraph_val);
//             console.log (question_val);
//             console.log (answer_val);
//             var asking_obj ={"para":paragraph_val, "ques":question_val}
//             var loader = document.getElementsByClassName("loader")[0];
//             loader.style.display = "block";
    
//             fetch('http://127.0.0.1:5000/qa_processing', {
//                 method: 'post',
//                 headers: {
//                     'Content-Type': 'application/json'
//                   },
//                 body: JSON.stringify(asking_obj)
//             }).then(function(response) {
//                 return response.json();
//             }).then(function(data) {
//                 console.log(data);
//                 var loader = document.getElementsByClassName("loader")[0];
//                 loader.style.display = "none";
//                 answer_text= "";
//                 var keys = Object.keys(data);
//                 for(var i=0; i<keys.length; i++){
//                     var key = keys[i];
//                     answer_text += "Q : "+key +"\nA : "+data[key].toUpperCase()+"\n";
//                     console.log(key, data[key]);
//                 }
//                 document.getElementById("input_answer").value = answer_text;
    
//             });
//         });
//     }



    
//     var submit_btn= document.getElementsByClassName("qa_link_submit")[0];
//     if (submit_btn != undefined){
//         submit_btn.addEventListener('click', function (event) {
//             event.preventDefault();
            
//             var wiki_val = document.getElementById("wiki-link").value;
//             var question_val = document.getElementById("input_question").value;
//             var answer_val = document.getElementById("input_link_answer").value;
//             console.log (wiki_val);
//             console.log (question_val);
//             console.log (answer_val);
//             var asking_obj ={"wiki":wiki_val, "ques":question_val}
//             var loader = document.getElementsByClassName("loader")[0];
//             loader.style.display = "block";
    
//             fetch('http://127.0.0.1:5000/qa_link_processing', {
//                 method: 'post',
//                 headers: {
//                     'Content-Type': 'application/json'
//                   },
//                 body: JSON.stringify(asking_obj)
//             }).then(function(response) {
//                 return response.json();
//             }).then(function(data) {
//                 console.log(data);
//                 var loader = document.getElementsByClassName("loader")[0];
//                 loader.style.display = "none";
//                 answer_text= "";
//                 var keys = Object.keys(data);
//                 for(var i=1; i<keys.length; i++){
//                     var key = keys[i];
//                     answer_text += "Q : "+key +
//                     "\nA : "+data[key][0].toUpperCase()+
//                     "\n"+data[key][1]+
//                     "\n"+"-------------------------------------------"+
//                     "\n";
//                     console.log(key, data[key]);
//                 }
//                 document.getElementById("input_link_answer").value = answer_text;
    
//             });
//         });
//     }



// });

window.addEventListener('load', function () {

    var submit_btn= document.getElementsByClassName("qa_submit")[0];

    if (submit_btn != undefined){
        submit_btn.addEventListener('click', function (event) {
            event.preventDefault();
            
            var paragraph_val = document.getElementById("input_paragraph").value;
            var question_val = document.getElementById("input_question").value;
            var answer_val = document.getElementById("input_answer").value;
            console.log (paragraph_val);
            console.log (question_val);
            console.log (answer_val);
            var asking_obj ={"para":paragraph_val, "ques":question_val}
            var loader = document.getElementsByClassName("loader")[0];
            loader.style.display = "block";
        
    
            fetch('http://34.80.65.17:5000/qa_processing', {
                method: 'post',
                headers: {
                    'Content-Type': 'application/json'
                  },
                body: JSON.stringify(asking_obj)
            }).then(function(response) {
                return response.json();
            }).then(function(data) {
                console.log(data);
                var loader = document.getElementsByClassName("loader")[0];
                loader.style.display = "none";
                answer_text= "";
                var keys = Object.keys(data);
                for(var i=0; i<keys.length; i++){
                    var key = keys[i];
                    answer_text += "Q : "+key +"\nA : "+data[key].toUpperCase()+"\n";
                    console.log(key, data[key]);
                }
                document.getElementById("input_answer").value = answer_text;
    
            });
        });
    }



    
    var submit_btn= document.getElementsByClassName("qa_link_submit")[0];
    if (submit_btn != undefined){
        submit_btn.addEventListener('click', function (event) {
            event.preventDefault();
            
            var wiki_val = document.getElementById("wiki-link").value;
            var question_val = document.getElementById("input_question").value;
            var answer_val = document.getElementById("input_link_answer").value;
            console.log (wiki_val);
            console.log (question_val);
            console.log (answer_val);
            var asking_obj ={"wiki":wiki_val, "ques":question_val};
            var loader = document.getElementsByClassName("loader")[0];
            loader.style.display = "block";
        
    
            fetch('http://34.80.65.17:5000/qa_link_processing', {
                method: 'post',
                headers: {
                    'Content-Type': 'application/json'
                  },
                body: JSON.stringify(asking_obj)
            }).then(function(response) {
                return response.json();
            }).then(function(data) {
                console.log(data);
                var loader = document.getElementsByClassName("loader")[0];
                loader.style.display = "none";
                answer_text= "";
                var keys = Object.keys(data);
                for(var i=1; i<keys.length; i++){
                    var key = keys[i];
                    answer_text += "Q : "+key +
                    "\nA : "+data[key][0].toUpperCase()+
                    "\n"+data[key][1]+
                    "\n"+"-------------------------------------------"+
                    "\n";
                    console.log(key, data[key]);
                }
                document.getElementById("input_link_answer").value = answer_text;
    
            });
        });
    }



});


// document.getElementById().style.visibility = "visible";
// export IMAGE_FAMILY="tf2-latest-gpu"
// export ZONE="asia-east1-c"
// export INSTANCE_NAME="natural-questions-answering"
  
// gcloud compute instances create $INSTANCE_NAME \
//   --zone=$ZONE \
//   --image-family=$IMAGE_FAMILY \
//   --image-project=deeplearning-platform-release \
//   --maintenance-policy=TERMINATE \
//   --accelerator="type=nvidia-tesla-v100,count=1" \
//   --metadata="install-nvidia-driver=True" \
//   --boot-disk-size=200GB \
//   --machine-type=n1-highmem-8 \
//   --tags=[http-server,https-server]
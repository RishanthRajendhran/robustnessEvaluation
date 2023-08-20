<h1>
    Evaluating robustness of large scale language models
</h1>
<h4>
    Rishanth Rajendhran
</h4>
<p>
    This is the repository for the research on robustness large-scale language models done with Professor Ana Marasovic at The University of Utah.
</p>
<h5>
    LLMs supported
</h5>
<ul>
    <li>
        FLAN-T5 (Standard decoding and Self Consistency)
    </li>
    <li>    
        Llama (Self Consistency only)
    </li>
    <li>
        Alpaca (Self Consistency only)
    </li>
    <li>
        MPT (Self Consistency only)
    </li>
</ul>
<h5>
    Files
</h5>
<ul>
    <li>
        <h5>
            test.py
        <h5>
        <p>
            This file is used to perfom inference using transformers pipeline
        </p>
        <h5>
            Important Note
        </h5>
        <p>
            All input files should be JSON files in the CondaQA format
            <br/>
<pre>
{
    "sentence1": <Passage>,
    "sentence2": <Question>,
    "label": <GoldLabel>, [Only for train instances]
    "explanation": <CoTExplanation> [Only for train instances when performing elicitive prompting]
}
</pre>
        </p>
        <h5>
            Usage
        </h5>
        <pre>
    usage: test.py [-h] -train TRAINFILES [TRAINFILES ...] -test TESTFILES
            [TESTFILES ...] [-isTrainDir] [-isTestDir]
            [-promptType {1,2,3,4,5,6}] [-bestPromptType {1,2,3}]
            [-zeroShot] -dataset
            {condaqa,boolq,ropes,drop,quoref,mctaco,imdb,matres,perspectrum,udparsing}
            [-model {flant5,mpt}] [-modelSize MODELSIZE]
            [-trainPattern TRAINPATTERN] [-testPattern TESTPATTERN]
            [-selfConsistency] [-noCoT]

    optional arguments:
    -h, --help            show this help message and exit
    -train TRAINFILES [TRAINFILES ...], --trainFiles TRAINFILES [TRAINFILES ...]
    -test TESTFILES [TESTFILES ...], --testFiles TESTFILES [TESTFILES ...]
    -isTrainDir, --isTrainDirectory
                            Booleaan flag to indicate if the -train input is a
                            directory path
    -isTestDir, --isTestDirectory
                            Booleaan flag to indicate if the -test input is a
                            directory path
    -promptType {1,2,3,4,5,6}
    -bestPromptType {1,2,3}
                            When promptType is set as 4/5, bestPromptType defines
                            the sub-prompt type
    -zeroShot
    -dataset {condaqa,boolq,ropes,drop,quoref,mctaco,imdb,matres,perspectrum,udparsing}
    -model {flant5,mpt}   Name of model to use for inference
    -modelSize MODELSIZE
    -trainPattern TRAINPATTERN
                            RegEx pattern for json file names in the train
                            directory that need to be used
    -testPattern TESTPATTERN
                            RegEx pattern for json file names in the test
                            directory that need to be merged
    -selfConsistency      Boolean flag to enable self consistency mode
    -noCoT                Boolean flag to indicate no-Chain-of-Thought
                            inferencing
</pre>
        <h5>
            Flags
        </h5>
        <ul>
            <li>
                <b>
                    train [REQUIRED]
                </b>
                <p>
                    This flag is used to provide the path to one of the following:
                    <ol>
                        <li>
                            Train file
                        </li>
                        <li>
                            Train files (as a space separated list)
                        </li>
                        <li>
                            Directory containing train files
                        </li>
                    </ol>
                </p>
            </li>
            <li>
                <b>
                    test [REQUIRED]
                </b>
                <p>
                    This flag is used to provide the path to one of the following:
                    <ol>
                        <li>
                            Test file
                        </li>
                        <li>
                            Test files (as a space separated list)
                        </li>
                        <li>
                            Directory containing test files
                        </li>
                    </ol>
                </p>
            </li>
            <li>
                <b>
                    isTrainDir [OPTIONAL]
                </b>
                <p>
                    Boolean flag to indicate that argument to -train is a directory path containing train files
                </p>
            </li>
            <li>
                <b>
                    isTestDir [OPTIONAL]
                </b>
                <p>
                    Boolean flag to indicate that argument to -test is a directory path containing test files
                </p>
            </li>
            <li>
                <b>
                    trainPattern [OPTIONAL]
                </b>
                <p>
                    RegEx pattern for json files in the train directory path that need to be used
                    <br/>
                    Default: All files in the train directory path
                </p>
            </li>
            <li>
                <b>
                    testPattern [OPTIONAL]
                </b>
                <p>
                    RegEx pattern for json files in the test directory path that need to be used
                    <br/>
                    Default: All files in the test directory path
                </p>
            </li>
            <li>
                <b>
                    promptType [OPTIONAL]
                </b>
                <p>
                    Type of prompt to use
                    Default: 1
                </p>
            </li>
            <li>
                <b>
                    bestPromptType [OPTIONAL]
                </b>
                <p>
                    Sub-prompt type for prompt types 4 and 5
                    Default: 1
                </p>
            </li>
            <li>
                <b>
                    zeroShot [OPTIONAL]
                </b>
                <p>
                    Boolean flag to enable zero-shot evaluation
                </p>
            </li>
            <li>
                <b>
                    dataset [REQUIRED]
                </b>
                <p>
                    Name of dataset to use
                </p>
            </li>
            <li>
                <b>
                    model [OPTIONAL]
                </b>
                <p>
                    Type of model to use for inference
                    Default: "flant5"
                </p>
            </li>
            <li>
                <b>
                    modelSize [OPTIONAL]
                </b>
                <p>
                    Size of model to use (when model is "flant5")
                    Default: "xxl"
                </p>
            </li>
            <li>
                <b>
                    selfConsistency [OPTIONAL]
                </b>
                <p>
                    Boolean flag to enable self-consistency prompting 
                </p>
            </li>
            <li>
                <b>
                    noCoT [OPTIONAL]
                </b>
                <p>
                    Boolean flag to enable standard prompting
                </p>
            </li>
        </ul>
        <h5>
            Sample Usage
        </h5>
        <p>
            Perform CoT prompting for CondaQA dataset using FLANT5-XXL model using all train files in the path "./samples/CondaQA/trainTest/train" and the test file "./samples/CondaQA/trainTest/test/merged/condaqa_test_merged.json"
        </p>
        <pre>
python3 test.py -model flant5 -modelSize xxl -dataset condaqa -promptType 1 -train ./samples/CondaQA/trainTest/train -test ./samples/CondaQA/trainTest/test/merged/condaqa_test_merged.json -isTrainDir
        </pre>
    </li>
    <li>
        <h5>
            testConsistency.py
        </h5>
        <p>
            This file is used to calculate performance metrics such as accuracy, consistency, f1 score and exact match scores for predictions generated using test.py 
        </p>
        <h5>
            Important Note
        </h5>
        <p>
            All input files should be JSON files in the CondaQA format
            <br/>
<pre>
{
    "PassageID": <PassageID>,
    "QuestionID": <QuestionID>,
    "isOriginal": <True/False>,
    "SampleID": <SampleID>, [Only when performing self-consistency prompting]
    "sentence1": <Passage>,
    "sentence2": <Question>,
    "output": <Prediction>,
    "label": <GoldLabel>, [Only for train instances],
    "PassageEditID": <PassageEditID>, [Only for CondaQA dataset]
}
</pre>
            <br/>
            Note that every instance in the predictions file should have a "isOriginal" attribute indicating whether it is an original instance or perturbed instance. When not performing contrastive evaluation, set the value True for all instances in your predictions file.
            <br/>
            When performing self-consistency prompting, SampleID needs to be provided. All predictions which are essentially different outputs based on different reasoning paths sampled for the same question need to have the same SampleID
            <br/>
            Note that the "output" field expects predictions of the form: 
<pre>
"[Explanation]. The answer is [FinalAnswer]."
</pre>
            Explanation and punctuation marks are optional.
        </p>
        <h5>
            Usage
        </h5>
<pre>
usage: testConsistency.py [-h] -out OUTPUTFILES [OUTPUTFILES ...] [-isDir]
                          [-concise] [-summaryOnly] -dataset
                          {condaqa,boolq,ropes,drop,quoref,mctaco,imdb,matres,perspectrum,udparsing}
                          [-selfConsistency] [-f1 F1THRESHOLD]
                          [-pattern PATTERN]

optional arguments:
  -h, --help            show this help message and exit
  -out OUTPUTFILES [OUTPUTFILES ...], --outputFiles OUTPUTFILES [OUTPUTFILES ...]
                        List of output file names/Path to directory containing
                        output json files (Need to set -isDir flag)
  -isDir, --isDirectory
                        Booleaan flag to indicate if the -out input is a
                        directory path
  -concise              Boolean flag to indicate if outputs need to be concise
  -summaryOnly          Booleaan flag to indicate if only summary needs to be
                        printed
  -dataset {condaqa,boolq,ropes,drop,quoref,mctaco,imdb,matres,perspectrum,udparsing}
  -selfConsistency      Booleaan flag to compute self-consistency
  -f1 F1THRESHOLD, --f1Threshold F1THRESHOLD
                        F1 Threshold to use for evaluation (between 0 and 1
                        only)
  -pattern PATTERN      RegEx pattern for json file names in the output
                        directory that need to be evaluated
</pre>
        <h5>
            Flags
        </h5>
        <ul>
            <li>
                <b>
                    out [REQUIRED]
                </b>
                <p>
                    This flag is used to provide the path to one of the following:
                    <ol>
                        <li>
                            Output file
                        </li>
                        <li>
                            Output files (as a space separated list)
                        </li>
                        <li>
                            Directory containing Output files
                        </li>
                    </ol>
                </p>
            </li>
            <li>
                <b>
                    isDir [OPTIONAL]
                </b>
                <p>
                    Boolean flag to indicate that argument to -out is a directory path containing output files
                </p>
            </li>
            <li>
                <b>
                    pattern [OPTIONAL]
                </b>
                <p>
                    RegEx pattern for json files in the output directory path that need to be used
                    <br/>
                    Default: All files in the output directory path
                </p>
            </li>
            <li>
                <b>
                    dataset [REQUIRED]
                </b>
                <p>
                    Name of dataset to use
                </p>
            </li>
            <li>
                <b>
                    selfConsistency [OPTIONAL]
                </b>
                <p>
                    Boolean flag to enable self-consistency prompting evaluation
                </p>
            </li>
            <li>
                <b>
                    f1 [OPTIONAL]
                </b>
                <p>
                    Threshold for f1 score to mark a prediction as correct
                    <br/>
                    Default: 0.8
                </p>
            </li>
            <li>
                <b>
                    concise [OPTIONAL]
                </b>
                <p>
                    Boolean flag to skip printing detailed information
                </p>
            </li>
            <li>
                <b>
                    summaryOnly [OPTIONAL]
                </b>
                <p>
                    Boolean flag to print only final summary across test files
                </p>
            </li>
        </ul>
        <h5>
            Sample Usage
        </h5>
        <p>
            Print overall summary of performance of model with CoT prompting for CondaQA dataset using all output files in the path "./testOuts/consistency/condaqa/merged/marked/"
        </p>
        <pre>
python3 testConsistency.py -out ./testOuts/consistency/condaqa/merged/marked/ -isDir -summaryOnly -dataset condaqa
        </pre>
    </li> 
    <li>
        <h6>
            pickSamples.py
        </h6>
    </li>  
    <li>
        <h6>
            prettyPrint.py
        </h6>
    </li> 
    <li>
        <h6>
            parseTxtToJson.py
        </h6>
    </li>  
    <li>
        <h6>
            demarcateOriginalFromPerturbed.py
        </h6>
    </li>
    <li>
        <h6>
            generateTrainTest.py
        </h6>
    </li>  
    <li>
        <h6>
            printPrompts.py
        </h6>
        <p>
            This file is used to print prompt samples
        </p>
        <h5>
            Important Note
        </h5>
        <p>
            All input files should be JSON files in the CondaQA format
            <br/>
<pre>
{
    "sentence1": <Passage>,
    "sentence2": <Question>,
    "label": <GoldLabel>, [Only for train instances]
    "explanation": <CoTExplanation> [Only for train instances when performing elicitive prompting]
}
</pre>
        </p>
        <h5>
            Usage
        </h5>
        <pre>
usage: printPrompts.py [-h] -train TRAINFILES [TRAINFILES ...] -test TESTFILES
                       [TESTFILES ...] [-isTrainDir] [-isTestDir]
                       [-promptType {1,2,3,4,5,6}] [-bestPromptType {1,2,3}]
                       [-zeroShot] -dataset
                       {condaqa,boolq,ropes,drop,quoref,mctaco,imdb,matres,perspectrum,udparsing}
                       [-trainPattern TRAINPATTERN] [-testPattern TESTPATTERN]
                       [-selfConsistency] [-noCoT]

optional arguments:
  -h, --help            show this help message and exit
  -train TRAINFILES [TRAINFILES ...], --trainFiles TRAINFILES [TRAINFILES ...]
  -test TESTFILES [TESTFILES ...], --testFiles TESTFILES [TESTFILES ...]
  -isTrainDir, --isTrainDirectory
                        Booleaan flag to indicate if the -train input is a
                        directory path
  -isTestDir, --isTestDirectory
                        Booleaan flag to indicate if the -test input is a
                        directory path
  -promptType {1,2,3,4,5,6}
  -bestPromptType {1,2,3}
                        When promptType is set as 4/5, bestPromptType defines
                        the sub-prompt type
  -zeroShot
  -dataset {condaqa,boolq,ropes,drop,quoref,mctaco,imdb,matres,perspectrum,udparsing}
  -trainPattern TRAINPATTERN
                        RegEx pattern for json file names in the train
                        directory that need to be used
  -testPattern TESTPATTERN
                        RegEx pattern for json file names in the test
                        directory that need to be merged
  -selfConsistency      Boolean flag to enable self consistency mode
  -noCoT                Boolean flag to indicate no-Chain-of-Thought
                        inferencing
</pre>
        <h5>
            Flags
        </h5>
        <ul>
            <li>
                <b>
                    train [REQUIRED]
                </b>
                <p>
                    This flag is used to provide the path to one of the following:
                    <ol>
                        <li>
                            Train file
                        </li>
                        <li>
                            Train files (as a space separated list)
                        </li>
                        <li>
                            Directory containing train files
                        </li>
                    </ol>
                </p>
            </li>
            <li>
                <b>
                    test [REQUIRED]
                </b>
                <p>
                    This flag is used to provide the path to one of the following:
                    <ol>
                        <li>
                            Test file
                        </li>
                        <li>
                            Test files (as a space separated list)
                        </li>
                        <li>
                            Directory containing test files
                        </li>
                    </ol>
                </p>
            </li>
            <li>
                <b>
                    isTrainDir [OPTIONAL]
                </b>
                <p>
                    Boolean flag to indicate that argument to -train is a directory path containing train files
                </p>
            </li>
            <li>
                <b>
                    isTestDir [OPTIONAL]
                </b>
                <p>
                    Boolean flag to indicate that argument to -test is a directory path containing test files
                </p>
            </li>
            <li>
                <b>
                    trainPattern [OPTIONAL]
                </b>
                <p>
                    RegEx pattern for json files in the train directory path that need to be used
                    <br/>
                    Default: All files in the train directory path
                </p>
            </li>
            <li>
                <b>
                    testPattern [OPTIONAL]
                </b>
                <p>
                    RegEx pattern for json files in the test directory path that need to be used
                    <br/>
                    Default: All files in the test directory path
                </p>
            </li>
            <li>
                <b>
                    promptType [OPTIONAL]
                </b>
                <p>
                    Type of prompt to use
                    Default: 1
                </p>
            </li>
            <li>
                <b>
                    bestPromptType [OPTIONAL]
                </b>
                <p>
                    Sub-prompt type for prompt types 4 and 5
                    Default: 1
                </p>
            </li>
            <li>
                <b>
                    zeroShot [OPTIONAL]
                </b>
                <p>
                    Boolean flag to enable zero-shot evaluation
                </p>
            </li>
            <li>
                <b>
                    dataset [REQUIRED]
                </b>
                <p>
                    Name of dataset to use
                </p>
            </li>
            <li>
                <b>
                    selfConsistency [OPTIONAL]
                </b>
                <p>
                    Boolean flag to enable self-consistency prompting 
                </p>
            </li>
            <li>
                <b>
                    noCoT [OPTIONAL]
                </b>
                <p>
                    Boolean flag to enable standard prompting
                </p>
            </li>
        </ul>
        <h5>
            Sample Usage
        </h5>
        <p>
            Print CoT prompts for CondaQA dataset using all train files in the path "./samples/CondaQA/trainTest/train" and the test file "./samples/CondaQA/trainTest/test/merged/condaqa_test_merged.json"
        </p>
        <pre>
python3 printPrompts.py -dataset condaqa -promptType 1 -train ./samples/CondaQA/trainTest/train -test ./samples/CondaQA/trainTest/test/merged/condaqa_test_merged.json -isTrainDir
</pre>
    </li>  
    <li>
        <h6>
            sampleStats.py
        </h6>
    </li>
    <li>
        <h6>
            splitJson.py
        </h6>
    </li> 
    <li>
        <h6>
            mergeJson.py
        </h6>
    </li>     
</ul>
// Modifying this comment will cause the next execution of LBJava to overwrite this file.
// F1B88000000000000000D4E81CA028044154F75ED251C2AF0863605B80929A6F1FC1F6A393D83CB9766F9F95016BDB70EE935BD8640A020EA10260145A7F941061F6D737CAEE4D757F496DB027F9D8A0EA0E8688314ACD9BD18570144A1DA6B4E132D930B44C10A36F2DD7F7029CC02F7248B14D5EC2B2958F15C66FB40E7ED31B223A5564F7CE2DA53D974C896A4140BD79A0B15AB1BB835EBFAFD0FB9447B1BC000000

package edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.ee;

import edu.illinois.cs.cogcomp.lbjava.classify.*;
import edu.illinois.cs.cogcomp.lbjava.infer.*;
import edu.illinois.cs.cogcomp.lbjava.io.IOUtilities;
import edu.illinois.cs.cogcomp.lbjava.learn.*;
import edu.illinois.cs.cogcomp.lbjava.parse.*;
import edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.LearningObj;
import edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.ParamLBJ;




public class ee_perceptron extends SparseNetworkLearner
{
  private static java.net.URL _lcFilePath;
  private static java.net.URL _lexFilePath;

  static
  {
    _lcFilePath = ee_perceptron.class.getResource("ee_perceptron.lc");
    _lexFilePath = ee_perceptron.class.getResource("ee_perceptron.lex");
  }

  private static void loadInstance()
  {
    if (instance == null)
    {
      if (_lcFilePath == null)
      {
        throw new RuntimeException("Can't locate ee_perceptron.lc in the class path.");
      }
      if (_lexFilePath == null)
      {
        throw new RuntimeException("Can't locate ee_perceptron.lc in the class path.");
      }
      instance = (ee_perceptron) Learner.readLearner(_lcFilePath);
      instance.readLexiconOnDemand(_lexFilePath);
    }
  }

  public static Parser getParser() { return null; }
  public static Parser getTestParser() { return null; }

  public static boolean isTraining;
  public static ee_perceptron instance;

  public static ee_perceptron getInstance()
  {
    loadInstance();
    return instance;
  }

  private ee_perceptron(boolean b)
  {
    super(new SparseAveragedPerceptron(ParamLBJ.eeLearningRate, 0, ParamLBJ.eeThickness));
    containingPackage = "edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.ee";
    name = "ee_perceptron";
    setEncoding(null);
    setLabeler(new Label());
    setExtractor(new AllFeatures());
    isClone = false;
  }

  public static TestingMetric getTestingMetric() { return null; }


  private boolean isClone;

  public void unclone() { isClone = false; }

  public ee_perceptron()
  {
    super("edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.ee.ee_perceptron");
    isClone = true;
  }

  public ee_perceptron(String modelPath, String lexiconPath) { this(new Parameters(), modelPath, lexiconPath); }

  public ee_perceptron(Parameters p, String modelPath, String lexiconPath) {
    super(p);
    try {
      lcFilePath = new java.net.URL("file:" + modelPath);
      lexFilePath = new java.net.URL("file:" + lexiconPath);
    }
    catch (Exception e) {
      System.err.println("ERROR: Can't create model or lexicon URL: " + e);
      e.printStackTrace();
      System.exit(1);
    }

    java.io.File modelfile = new java.io.File(modelPath);

    if (modelfile.exists()) {
      System.out.println("Model file read from "+modelfile.getAbsolutePath());
      readModel(lcFilePath);
      readLexiconOnDemand(lexFilePath);
    }
    else if (IOUtilities.existsInClasspath(ee_perceptron.class, modelPath)) {
      System.out.println("Model file "+modelfile.getAbsolutePath()+" located in a jar file");
      readModel(IOUtilities.loadFromClasspath(ee_perceptron.class, modelPath));
      readLexiconOnDemand(IOUtilities.loadFromClasspath(ee_perceptron.class, lexiconPath));
    }
    else {
      containingPackage = "edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.ee";
      name = "ee_perceptron";
      setLabeler(new Label());
      setExtractor(new AllFeatures());
    }

    isClone = false;
  }

  public String getInputType() { return "edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.LearningObj"; }
  public String getOutputType() { return "discrete"; }

  public void learn(Object example)
  {
    if (isClone)
    {
      if (!(example instanceof LearningObj || example instanceof Object[]))
      {
        String type = example == null ? "null" : example.getClass().getName();
        System.err.println("Classifier 'ee_perceptron(LearningObj)' defined on line 20 of ee_perceptron.lbj received '" + type + "' as input.");
        new Exception().printStackTrace();
        System.exit(1);
      }

      loadInstance();
      instance.learn(example);
      return;
    }

    if (example instanceof Object[])
    {
      Object[] a = (Object[]) example;
      if (a[0] instanceof int[])
      {
        super.learn((int[]) a[0], (double[]) a[1], (int[]) a[2], (double[]) a[3]);
        return;
      }
    }

    super.learn(example);
  }

  public void learn(Object[] examples)
  {
    if (isClone)
    {
      if (!(examples instanceof LearningObj[] || examples instanceof Object[][]))
      {
        String type = examples == null ? "null" : examples.getClass().getName();
        System.err.println("Classifier 'ee_perceptron(LearningObj)' defined on line 20 of ee_perceptron.lbj received '" + type + "' as input.");
        new Exception().printStackTrace();
        System.exit(1);
      }

      loadInstance();
      instance.learn(examples);
      return;
    }

    super.learn(examples);
  }

  public Feature featureValue(Object __example)
  {
    if (isClone)
    {
      if (!(__example instanceof LearningObj || __example instanceof Object[]))
      {
        String type = __example == null ? "null" : __example.getClass().getName();
        System.err.println("Classifier 'ee_perceptron(LearningObj)' defined on line 20 of ee_perceptron.lbj received '" + type + "' as input.");
        new Exception().printStackTrace();
        System.exit(1);
      }

      loadInstance();
      return instance.featureValue(__example);
    }

    if (__example instanceof Object[])
    {
      Object[] a = (Object[]) __example;
      if (a[0] instanceof int[])
        return super.featureValue((int[]) a[0], (double[]) a[1]);
    }

    Feature __result;
    __result = super.featureValue(__example);
    return __result;
  }

  public FeatureVector classify(Object __example)
  {
    return new FeatureVector(featureValue(__example));
  }

  public String discreteValue(Object __example)
  {
    return featureValue(__example).getStringValue();
  }

  public FeatureVector[] classify(Object[] examples)
  {
    if (isClone)
    {
      if (!(examples instanceof LearningObj[] || examples instanceof Object[][]))
      {
        String type = examples == null ? "null" : examples.getClass().getName();
        System.err.println("Classifier 'ee_perceptron(LearningObj)' defined on line 20 of ee_perceptron.lbj received '" + type + "' as input.");
        new Exception().printStackTrace();
        System.exit(1);
      }

      loadInstance();
      return instance.classify(examples);
    }

    FeatureVector[] result = super.classify(examples);
    return result;
  }

  public static void main(String[] args)
  {
    String testParserName = null;
    String testFile = null;
    Parser testParser = getTestParser();

    try
    {
      if (!args[0].equals("null"))
        testParserName = args[0];
      if (args.length > 1) testFile = args[1];

      if (testParserName == null && testParser == null)
      {
        System.err.println("The \"testFrom\" clause was not used in the learning classifier expression that");
        System.err.println("generated this classifier, so a parser and input file must be specified.\n");
        throw new Exception();
      }
    }
    catch (Exception e)
    {
      System.err.println("usage: edu.illinois.cs.cogcomp.nlp.classifier.lbj.perceptron.ee.ee_perceptron \\");
      System.err.println("           <parser> <input file> [<null label> [<null label> ...]]\n");
      System.err.println("     * <parser> must be the fully qualified class name of a Parser, or \"null\"");
      System.err.println("       to use the default as specified by the \"testFrom\" clause.");
      System.err.println("     * <input file> is the relative or absolute path of a file, or \"null\" to");
      System.err.println("       use the parser arguments specified by the \"testFrom\" clause.  <input");
      System.err.println("       file> can also be non-\"null\" when <parser> is \"null\" (when the parser");
      System.err.println("       specified by the \"testFrom\" clause has a single string argument");
      System.err.println("       constructor) to use an alternate file.");
      System.err.println("     * A <null label> is a label (or prediction) that should not count towards");
      System.err.println("       overall precision and recall assessments.");
      System.exit(1);
    }

    if (testParserName == null && testFile != null && !testFile.equals("null"))
      testParserName = testParser.getClass().getName();
    if (testParserName != null)
      testParser = edu.illinois.cs.cogcomp.lbjava.util.ClassUtils.getParser(testParserName, new Class[]{ String.class }, new String[]{ testFile });
    ee_perceptron classifier = new ee_perceptron();
    TestDiscrete tester = new TestDiscrete();
    for (int i = 2; i < args.length; ++i)
      tester.addNull(args[i]);
    TestDiscrete.testDiscrete(tester, classifier, classifier.getLabeler(), testParser, true, 0);
  }

  public int hashCode() { return "ee_perceptron".hashCode(); }
  public boolean equals(Object o) { return o instanceof ee_perceptron; }

  public java.lang.String discreteValue(int[] a0, double[] a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.discreteValue(a0, a1);
    }

    return super.discreteValue(a0, a1);
  }

  public void learn(int[] a0, double[] a1, int[] a2, double[] a3)
  {
    if (isClone)
    {
      loadInstance();
      instance.learn(a0, a1, a2, a3);
      return;
    }

    super.learn(a0, a1, a2, a3);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.Feature featureValue(int[] a0, double[] a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.featureValue(a0, a1);
    }

    return super.featureValue(a0, a1);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector classify(int[] a0, double[] a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.classify(a0, a1);
    }

    return super.classify(a0, a1);
  }

  public int getNumExamples()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getNumExamples();
    }

    return super.getNumExamples();
  }

  public int getNumFeatures()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getNumFeatures();
    }

    return super.getNumFeatures();
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.LinearThresholdUnit getBaseLTU()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getBaseLTU();
    }

    return super.getBaseLTU();
  }

  public edu.illinois.cs.cogcomp.core.datastructures.vectors.OVector getNetwork()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getNetwork();
    }

    return super.getNetwork();
  }

  public boolean isUsingConjunctiveLabels()
  {
    if (isClone)
    {
      loadInstance();
      return instance.isUsingConjunctiveLabels();
    }

    return super.isUsingConjunctiveLabels();
  }

  public void setConjunctiveLabels(boolean a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setConjunctiveLabels(a0);
      return;
    }

    super.setConjunctiveLabels(a0);
  }

  public java.lang.Object getLTU(int a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.getLTU(a0);
    }

    return super.getLTU(a0);
  }

  public void setParameters(edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner.Parameters a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setParameters(a0);
      return;
    }

    super.setParameters(a0);
  }

  public void setLTU(edu.illinois.cs.cogcomp.lbjava.learn.LinearThresholdUnit a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setLTU(a0);
      return;
    }

    super.setLTU(a0);
  }

  public void setLabeler(edu.illinois.cs.cogcomp.lbjava.classify.Classifier a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setLabeler(a0);
      return;
    }

    super.setLabeler(a0);
  }

  public void setExtractor(edu.illinois.cs.cogcomp.lbjava.classify.Classifier a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setExtractor(a0);
      return;
    }

    super.setExtractor(a0);
  }

  public void setNetworkLabel(int a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setNetworkLabel(a0);
      return;
    }

    super.setNetworkLabel(a0);
  }

  public void doneLearning()
  {
    if (isClone)
    {
      loadInstance();
      instance.doneLearning();
      return;
    }

    super.doneLearning();
  }

  public void doneWithRound()
  {
    if (isClone)
    {
      loadInstance();
      instance.doneWithRound();
      return;
    }

    super.doneWithRound();
  }

  public void forget()
  {
    if (isClone)
    {
      loadInstance();
      instance.forget();
      return;
    }

    super.forget();
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet scores(int[] a0, double[] a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.scores(a0, a1);
    }

    return super.scores(a0, a1);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet scores(java.lang.Object a0, java.util.Collection a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.scores(a0, a1);
    }

    return super.scores(a0, a1);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet scores(int[] a0, double[] a1, java.util.Collection a2)
  {
    if (isClone)
    {
      loadInstance();
      return instance.scores(a0, a1, a2);
    }

    return super.scores(a0, a1, a2);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.Feature valueOf(int[] a0, double[] a1, java.util.Collection a2)
  {
    if (isClone)
    {
      loadInstance();
      return instance.valueOf(a0, a1, a2);
    }

    return super.valueOf(a0, a1, a2);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.Feature valueOf(java.lang.Object a0, java.util.Collection a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.valueOf(a0, a1);
    }

    return super.valueOf(a0, a1);
  }

  public void write(edu.illinois.cs.cogcomp.core.datastructures.vectors.ExceptionlessOutputStream a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.write(a0);
      return;
    }

    super.write(a0);
  }

  public void write(java.io.PrintStream a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.write(a0);
      return;
    }

    super.write(a0);
  }

  public void read(edu.illinois.cs.cogcomp.core.datastructures.vectors.ExceptionlessInputStream a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.read(a0);
      return;
    }

    super.read(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.Learner.Parameters getParameters()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getParameters();
    }

    return super.getParameters();
  }

  public void initialize(int a0, int a1)
  {
    if (isClone)
    {
      loadInstance();
      instance.initialize(a0, a1);
      return;
    }

    super.initialize(a0, a1);
  }

  public java.lang.String discreteValue(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.discreteValue(a0);
    }

    return super.discreteValue(a0);
  }

  public double realValue(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.realValue(a0);
    }

    return super.realValue(a0);
  }

  public double realValue(int[] a0, double[] a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.realValue(a0, a1);
    }

    return super.realValue(a0, a1);
  }

  public void learn(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector[] a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.learn(a0);
      return;
    }

    super.learn(a0);
  }

  public void learn(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.learn(a0);
      return;
    }

    super.learn(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.Feature featureValue(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.featureValue(a0);
    }

    return super.featureValue(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector[] classify(java.lang.Object[][] a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.classify(a0);
    }

    return super.classify(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector classify(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.classify(a0);
    }

    return super.classify(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector[] classify(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector[] a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.classify(a0);
    }

    return super.classify(a0);
  }

  public void setParameters(edu.illinois.cs.cogcomp.lbjava.learn.Learner.Parameters a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setParameters(a0);
      return;
    }

    super.setParameters(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet scores(java.lang.Object a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.scores(a0);
    }

    return super.scores(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet scores(edu.illinois.cs.cogcomp.lbjava.classify.FeatureVector a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.scores(a0);
    }

    return super.scores(a0);
  }

  public java.lang.Object[] getExampleArray(java.lang.Object a0)
  {
    if (isClone)
    {
      loadInstance();
      return instance.getExampleArray(a0);
    }

    return super.getExampleArray(a0);
  }

  public java.lang.Object[] getExampleArray(java.lang.Object a0, boolean a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.getExampleArray(a0, a1);
    }

    return super.getExampleArray(a0, a1);
  }

  public void setLexicon(edu.illinois.cs.cogcomp.lbjava.learn.Lexicon a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setLexicon(a0);
      return;
    }

    super.setLexicon(a0);
  }

  public void readLexiconOnDemand(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.readLexiconOnDemand(a0);
      return;
    }

    super.readLexiconOnDemand(a0);
  }

  public void readLexiconOnDemand(java.net.URL a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.readLexiconOnDemand(a0);
      return;
    }

    super.readLexiconOnDemand(a0);
  }

  public void unsetLossFlag()
  {
    if (isClone)
    {
      loadInstance();
      instance.unsetLossFlag();
      return;
    }

    super.unsetLossFlag();
  }

  public void setCandidates(int a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setCandidates(a0);
      return;
    }

    super.setCandidates(a0);
  }

  public void setLossFlag()
  {
    if (isClone)
    {
      loadInstance();
      instance.setLossFlag();
      return;
    }

    super.setLossFlag();
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.Classifier getLabeler()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getLabeler();
    }

    return super.getLabeler();
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.Classifier getExtractor()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getExtractor();
    }

    return super.getExtractor();
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.Lexicon getLexicon()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getLexicon();
    }

    return super.getLexicon();
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.Lexicon getCurrentLexicon()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getCurrentLexicon();
    }

    return super.getCurrentLexicon();
  }

  public void setLabelLexicon(edu.illinois.cs.cogcomp.lbjava.learn.Lexicon a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setLabelLexicon(a0);
      return;
    }

    super.setLabelLexicon(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.Lexicon getLabelLexicon()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getLabelLexicon();
    }

    return super.getLabelLexicon();
  }

  public void setEncoding(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setEncoding(a0);
      return;
    }

    super.setEncoding(a0);
  }

  public void setModelLocation(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setModelLocation(a0);
      return;
    }

    super.setModelLocation(a0);
  }

  public void setModelLocation(java.net.URL a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setModelLocation(a0);
      return;
    }

    super.setModelLocation(a0);
  }

  public java.net.URL getModelLocation()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getModelLocation();
    }

    return super.getModelLocation();
  }

  public void setLexiconLocation(java.net.URL a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setLexiconLocation(a0);
      return;
    }

    super.setLexiconLocation(a0);
  }

  public void setLexiconLocation(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.setLexiconLocation(a0);
      return;
    }

    super.setLexiconLocation(a0);
  }

  public java.net.URL getLexiconLocation()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getLexiconLocation();
    }

    return super.getLexiconLocation();
  }

  public void countFeatures(edu.illinois.cs.cogcomp.lbjava.learn.Lexicon.CountPolicy a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.countFeatures(a0);
      return;
    }

    super.countFeatures(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.Lexicon getLexiconDiscardCounts()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getLexiconDiscardCounts();
    }

    return super.getLexiconDiscardCounts();
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.Learner emptyClone()
  {
    if (isClone)
    {
      loadInstance();
      return instance.emptyClone();
    }

    return super.emptyClone();
  }

  public void setReadLexiconOnDemand()
  {
    if (isClone)
    {
      loadInstance();
      instance.setReadLexiconOnDemand();
      return;
    }

    super.setReadLexiconOnDemand();
  }

  public edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet scoresAugmented(java.lang.Object a0, edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet a1)
  {
    if (isClone)
    {
      loadInstance();
      return instance.scoresAugmented(a0, a1);
    }

    return super.scoresAugmented(a0, a1);
  }

  public int getPrunedLexiconSize()
  {
    if (isClone)
    {
      loadInstance();
      return instance.getPrunedLexiconSize();
    }

    return super.getPrunedLexiconSize();
  }

  public void saveModel()
  {
    if (isClone)
    {
      loadInstance();
      instance.saveModel();
      return;
    }

    super.saveModel();
  }

  public void saveLexicon()
  {
    if (isClone)
    {
      loadInstance();
      instance.saveLexicon();
      return;
    }

    super.saveLexicon();
  }

  public void writeModel(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.writeModel(a0);
      return;
    }

    super.writeModel(a0);
  }

  public void writeLexicon(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.writeLexicon(a0);
      return;
    }

    super.writeLexicon(a0);
  }

  public void readModel(java.net.URL a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.readModel(a0);
      return;
    }

    super.readModel(a0);
  }

  public void readModel(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.readModel(a0);
      return;
    }

    super.readModel(a0);
  }

  public void readLexicon(java.lang.String a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.readLexicon(a0);
      return;
    }

    super.readLexicon(a0);
  }

  public void readLexicon(java.net.URL a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.readLexicon(a0);
      return;
    }

    super.readLexicon(a0);
  }

  public void readLabelLexicon(edu.illinois.cs.cogcomp.core.datastructures.vectors.ExceptionlessInputStream a0)
  {
    if (isClone)
    {
      loadInstance();
      instance.readLabelLexicon(a0);
      return;
    }

    super.readLabelLexicon(a0);
  }

  public edu.illinois.cs.cogcomp.lbjava.learn.Lexicon demandLexicon()
  {
    if (isClone)
    {
      loadInstance();
      return instance.demandLexicon();
    }

    return super.demandLexicon();
  }

  public void write(java.lang.String a0, java.lang.String a1)
  {
    if (isClone)
    {
      loadInstance();
      instance.write(a0, a1);
      return;
    }

    super.write(a0, a1);
  }

  public void read(java.lang.String a0, java.lang.String a1)
  {
    if (isClone)
    {
      loadInstance();
      instance.read(a0, a1);
      return;
    }

    super.read(a0, a1);
  }

  public void save()
  {
    if (isClone)
    {
      loadInstance();
      instance.save();
      return;
    }

    super.save();
  }

  public static class Parameters extends SparseNetworkLearner.Parameters
  {
    public Parameters() { super((SparseNetworkLearner.Parameters) new ee_perceptron(false).getParameters()); }
  }
}


package com.senzari.psl

/**
 * Created by qiusha on 2/24/15.
 */

import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries

import java.text.Normalizer
import java.util.regex.Pattern;
import org.apache.lucene.search.spell.LevensteinDistance;

/*
 * The first thing we need to do is initialize a ConfigBundle and a DataStore
 */

ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("senzari-basic-matching-example")

/* Uses H2 as a DataStore and stores it in a temp. directory by default */
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "senzari-basic-matching-example")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

System.err.println("dbpath = ${dbpath}")

/*
 * Now we can initialize a PSLModel, which is the core component of PSL.
 * The first constructor argument is the context in which the PSLModel is defined.
 * The second argument is the DataStore we will be using.
 */
PSLModel m = new PSLModel(this, data)

/*
 * We create three predicates in the model, giving their names and list of argument types
 */

//m.add predicate: "artistName" , types: [ArgumentType.UniqueID, ArgumentType.String]
m.add predicate: "trackTitle" , types: [ArgumentType.UniqueID, ArgumentType.String]
m.add predicate: "trackArtist" , types: [ArgumentType.UniqueID, ArgumentType.String]
m.add predicate: "trackAlbum" , types: [ArgumentType.UniqueID, ArgumentType.String]
m.add predicate: "trackYear" , types: [ArgumentType.UniqueID, ArgumentType.String]
m.add predicate: "trackSource" , types: [ArgumentType.UniqueID, ArgumentType.String]

m.add predicate: "sameTrack", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

/*
 * Now, we define a string similarity function bound to a predicate.
 * Note that we can use any implementation of ExternalFunction that acts on two swc strings!
 */
m.add function: "sameName" , implementation: new LevenshteinSimilarity()
//m.add function: "sameName" , implementation: new StringSimilarity()
m.add function: "sameYear" , implementation: new YearSimilarity()
m.add function: "sameSource", implementation: new SourceSimilarity()

/*
 * Having added all the predicates we need to represent our problem, we finally insert some rules into the model.
 * Rules are defined using a logical syntax. Uppercase letters are variables and the predicates used in the rules below
 * are those defined above. The character '&' denotes a conjunction wheres '>>' denotes a conclusion.
 * Each rule can be given a user defined weight or no weight is specified if it is learned.
 *
 * 'A ^ B' is a shorthand syntax for nonsymmetric(A,B), which means that in the grounding of the rule,
 * PSL does not ground the symmetric case.
 */
//m.add rule : ( artistName(A,AName) & artistName(B,BName) & (A ^ B) & sameName(AName,BName) ) >> sameArtist(A,B),  weight : 5

m.add rule : ( trackTitle(A,AName) & trackTitle(B,BName) & (A ^ B) & sameName(AName,BName)) >> sameTrack(A,B),  weight : 5
m.add rule : ( trackYear(A,AYear) & trackYear(B,BYear) & sameYear(AYear,BYear)) >> sameTrack(A,B),  weight : 3
m.add rule : (trackArtist(A, AArtist) & trackArtist(B, BArtist) & (A ^ B) & sameName(AArtist, BArtist)) >> sameTrack(A,B),  weight : 5
m.add rule : (trackAlbum(A, AAlbum) & trackAlbum(B, BAlbum) & (A ^ B) & sameName(AAlbum, BAlbum)) >> sameTrack(A,B),  weight : 3
m.add rule : (sameTrack(A,B) & trackSource(A, ASource) & trackSource(B, BSource)) >> ~sameSource(ASource, BSource), constraint : true
//m.add rule : (trackSource(A, ASource) & trackSource(B, BSource) & sameSource(ASource, BSource)) >> ~sameTrack(A,B), constraint : true


/* Now, we move on to defining rules with sets. Before we can use sets in rules, we have to define how we would like those sets
 * to be compared. For this we define the set comparison predicate 'sameFriends' which compares two sets of friends. For each
 * set comparison predicate, we need to specify the type of aggregator function to use, in this case its the Jaccard equality,
 * and the predicate which is used for comparison (which must be binary). Note that you can also define your own aggregator functions.
 */
//m.add setcomparison: "sameTracks" , using: SetComparison.Equality, on : sameTrack

/* Having defined a set comparison predicate, we can apply it in a rule. The body of the following rule is as above. However,
 * in the head, we use the 'sameFriends' set comparison to compare two sets defined using curly braces. To identify the elements
 * that are contained in the set, we can use object oriented syntax, where A.knows, denotes all those entities that are related to A
 * via the 'knows' relation, i.e the set { X | knows(A,X) }. The '+' operator denotes set union. We can also qualify a relation with
 * the 'inv' or 'inverse' keyword to denote its inverse.
 */
//m.add rule :  (sameArtist(A,B) & (A ^ B )) >> sameTracks( {A.artistHasTracks} , {B.artistHasTracks} ) , weight : 5

//m.add rule :  (sameTrack(A,B) & (A ^ B )) >> sameArtist( A, B) , weight : 5

/* Next, we define some constraints for our model. In this case, we restrict that each person can be aligned to at most one other person
 * in the other social network. To do so, we define two partial functional constraints where the latter is on the inverse.
 * We also say that samePerson must be symmetric, i.e., samePerson(p1, p2) == samePerson(p2, p1).
 */
//m.add PredicateConstraint.PartialFunctional , on : sameArtist
//m.add PredicateConstraint.PartialInverseFunctional , on : sameArtist
//m.add PredicateConstraint.Symmetric, on : sameArtist
m.add PredicateConstraint.PartialFunctional , on : sameTrack
m.add PredicateConstraint.PartialInverseFunctional , on : sameTrack
m.add PredicateConstraint.Symmetric, on : sameTrack

/*
 * Finally, we define a prior on the inference predicate samePerson. It says that we should assume two
 * people are not the samePerson with a little bit of weight. This can be overridden with evidence as defined
 * in the previous rules.
 */
m.add rule: ~sameTrack(A,B), weight: 1

println m;

def dir = '/data/proc/psl/testData/';
//def dir = '/Users/qiusha/dev/senzari/psl/testData/';
def partition = new Partition(0);

//def insert = data.getInserter(artistName, partition);
//InserterUtils.loadDelimitedData(insert, dir+"artistName");

insert = data.getInserter(trackTitle, partition);
InserterUtils.loadDelimitedData(insert, dir+"trackTitle");


insert = data.getInserter(trackArtist, partition);
InserterUtils.loadDelimitedData(insert, dir+"trackArtist");

insert = data.getInserter(trackAlbum, partition);
InserterUtils.loadDelimitedData(insert, dir+"trackAlbum");

insert = data.getInserter(trackYear, partition);
InserterUtils.loadDelimitedData(insert, dir+"trackYear");

insert = data.getInserter(trackSource, partition);
InserterUtils.loadDelimitedData(insert, dir+"trackSource");

Database db = data.getDatabase(partition, [TrackTitle, TrackArtist, TrackAlbum, TrackYear, TrackSource] as Set);
LazyMPEInference inferenceApp = new LazyMPEInference(m, db, config);
inferenceApp.mpeInference();
inferenceApp.close();

resultFile = new File("${dir}output")
if(resultFile.exists())
    resultFile.delete()
println "Inference results with hand-defined weights:"
for (GroundAtom atom : Queries.getAllAtoms(db, SameTrack))
    resultFile << atom.toString() + "\t" + atom.getValue() + "\n";

/*
class YearSimilarity implements ExternalFunction {

    @Override
    public int getArity() {
        return 2;
    }

    @Override
    public ArgumentType[] getArgumentTypes() {
        return [ArgumentType.String, ArgumentType.String].toArray();
    }

    @Override
    public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
        try {
            return Integer.parseInt(args[0].getValue()) == Integer.parseInt(args[1].getValue()) ? 1.0 : 0.0;
        }catch(Exception e){
            return 0.0
        }
    }

}

class SourceSimilarity implements ExternalFunction {

    @Override
    public int getArity() {
        return 2;
    }

    @Override
    public ArgumentType[] getArgumentTypes() {
        return [ArgumentType.String, ArgumentType.String].toArray();
    }

    @Override
    public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
        try {
            return args[0].getValue() == args[1].getValue() ? 1.0 : 0.0;
        }catch(Exception e){
            return 0.0
        }
    }

}
*/
class StringSimilarity implements ExternalFunction {

    @Override
    public int getArity() {
        return 2;
    }

    @Override
    public ArgumentType[] getArgumentTypes() {
        return [ArgumentType.String, ArgumentType.String].toArray();
    }

    @Override
    public double getValue(ReadOnlyDatabase db, GroundTerm... args) {
        def levenshtein = new LevensteinDistance();
        def score = levenshtein.getDistance(slugify(args[0].toString().replaceAll("\\s", "")), slugify(args[1].toString().replaceAll("\\s", "")));
        return score;
    }

    def slugify(String text) {
        Pattern NONLATIN = Pattern.compile("[^\\w:-]");
        Pattern WHITESPACE = Pattern.compile("[\\s]");

        String nowhitespace = WHITESPACE.matcher(text).replaceAll(":");
        String normalized = Normalizer.normalize(nowhitespace, Normalizer.Form.NFD);
        String slug = NONLATIN.matcher(normalized).replaceAll("");
        String result =  slug.replace(":", " ").replaceAll("[\\s]+", " ").toLowerCase(Locale.ENGLISH);
        return result;
    }
}

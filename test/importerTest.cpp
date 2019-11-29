#include <gtest/gtest.h>

#include "dataHandlers/dataHandler.h"
#include "dataHandlers/primitiveTypeArray.h"
#include "instructions/addPrimitiveType.h"
#include "instructions/multByConstParam.h"
#include "program/program.h"
#include "program/line.h"
#include "tpg/tpgVertex.h"
#include "tpg/tpgTeam.h"
#include "tpg/tpgAction.h"
#include "tpg/tpgEdge.h"
#include "tpg/tpgGraph.h"
#include "learn/learningParameters.h"

#include "importer/tpgGraphDotImporter.h"
#include "exporter/tpgGraphDotExporter.h"


class ImporterTest : public ::testing::Test {
public:
	ImporterTest() : tpg(NULL) {};

protected:
	const size_t size1{ 24 };
	std::vector<std::reference_wrapper<const DataHandlers::DataHandler>> vect;
	Instructions::Set set;
	Environment* e = NULL;
	std::vector<std::shared_ptr<Program::Program>> progPointers;
	Exporter::TPGGraphDotExporter* dotExporter = NULL;

	TPG::TPGGraph* tpg;
	std::vector<const TPG::TPGEdge*> edges;

	virtual void SetUp() {
		// Setup environment
		vect.push_back(*(new DataHandlers::PrimitiveTypeArray<double>((unsigned int)size1)));

		// Put a 1 in the dataHandler to make it easy to have non-zero return in Programs.
		((DataHandlers::PrimitiveTypeArray<double>&)vect.at(0).get()).setDataAt(typeid(PrimitiveType<double>), 0, 1.0);

		set.add(*(new Instructions::AddPrimitiveType<double>()));
		set.add(*(new Instructions::MultByConstParam<double, float>()));
		e = new Environment(set, vect, 8);
		tpg = new TPG::TPGGraph(*e);

		// Create 10 programs
		for (int i = 0; i < 8; i++) {
			progPointers.push_back(std::shared_ptr<Program::Program>(new Program::Program(*e)));
		}

		//add instructions to at least one program.
		for(int i = 0; i < 3; i++)
		{
			Program::Line & l = progPointers.at(0).get()->addNewLine();
			l.setInstructionIndex(0);
			l.setDestinationIndex(1);
			l.setParameter(0, 0.2f);
			l.setOperand(0,0,1);
		}

		// Create a TPG 
		// (T= Team, A= Action)
		// 
		//        .------. 
		//        v      |
		// T0---->T1---->T2     T4
		// |     /| \    |      |
		// v    / v  \   v      v
		// A0<-'  A1  `->A2     A3   A4
		// 
		// With four action and four teams
		// All Edges have a unique Program, except T1->A0 and T0->A0 which 
		// share the same program: progPointers.at(0)
		for (int i = 0; i < 4; i++) {
			tpg->addNewTeam();
		}
		for (int i = 0; i < 4; i++) {
			// Each action is linked to a team (and vice-versa)
			tpg->addNewAction(i);
			edges.push_back(&tpg->addNewEdge(*tpg->getVertices().at(i), *tpg->getVertices().back(), progPointers.at(i)));
		}

		// Add an additional Root Action
		tpg->addNewAction(4);

		// Add new Edges between teams
		edges.push_back(&tpg->addNewEdge(*tpg->getVertices().at(0), *tpg->getVertices().at(1), progPointers.at(4)));
		edges.push_back(&tpg->addNewEdge(*tpg->getVertices().at(1), *tpg->getVertices().at(2), progPointers.at(5)));

		// Add a cyclic edge
		edges.push_back(&tpg->addNewEdge(*tpg->getVertices().at(2), *tpg->getVertices().at(1), progPointers.at(6)));

		// Add new outgoing edge to one team
		edges.push_back(&tpg->addNewEdge(*tpg->getVertices().at(1), *tpg->getVertices().at(4), progPointers.at(0)));
		edges.push_back(&tpg->addNewEdge(*tpg->getVertices().at(1), *tpg->getVertices().at(6), progPointers.at(7)));

		// Check the characteristics
		ASSERT_EQ(tpg->getNbVertices(), 9);
		ASSERT_EQ(tpg->getEdges().size(), 9);
		ASSERT_EQ(tpg->getRootVertices().size(), 3);

		// Save the graph in a dot file.
		Exporter::TPGGraphDotExporter dotexporter ("exported_tpg.dot", *tpg);
		dotexporter.print();

	}

	virtual void TearDown() {
		delete tpg;
		delete e;
		delete (&(vect.at(0).get()));
		delete (&set.getInstruction(0));
		delete (&set.getInstruction(1));
	}
};

TEST_F(ImporterTest, Constructor) {
	Importer::TPGGraphDotImporter* dotImporter;
	ASSERT_NO_THROW(dotImporter = new Importer::TPGGraphDotImporter("exported_tpg.dot",*e)) << "The TPGGraphDotExporter could not be constructed with a valid file path.";

	ASSERT_NO_THROW(delete dotImporter;) << "TPGGraphDotExporter could not be deleted.";

	ASSERT_THROW(dotImporter = new Importer::TPGGraphDotImporter("XXX://INVALID_PATH", *e), std::runtime_error) << "The TPGGraphDotExplorer construction should fail with an invalid path.";
}

TEST_F(ImporterTest, importGraph)
{
	Importer::TPGGraphDotImporter dotImporter("exported_tpg.dot",*e);

	//assert that we can import a tpg graph from a file
	ASSERT_NO_THROW(dotImporter.importGraph()) << "The Graph import failed.";

	//actually import the model
	TPG::TPGGraph imported_model = std::move(dotImporter.importGraph());

	// Check the imported graph characteristics
	ASSERT_EQ(imported_model.getNbVertices(), 9) << "the wrong number of vertices have been created.";
	ASSERT_EQ(imported_model.getEdges().size(), 9) << "the wrong number of edges have been created.";
	ASSERT_EQ(imported_model.getRootVertices().size(), 3) << "the wrong number of root teams have been created.";

}

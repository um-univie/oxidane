use crate::molecule::Molecule;
use chemistry_consts::ElementProperties;
//use nauty_pet::prelude::*;

pub trait ToGml {
    fn to_gml(&self) -> String;
    fn to_gml_with_slice(&self, filter: &[usize], outer_filter: &[usize]) -> String;
}

impl<T: Molecule> ToGml for T {
    /// This function converts a `Molecule` to a gml file
    ///
    /// # Panics
    /// This function does not panic
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = Molecule3D::from_xyz("tests/ethane.xyz").unwrap().remove(0);
    /// let gml = molecule.to_gml();
    /// assert!(gml.contains("graph"));
    /// ```
    ///
    fn to_gml(&self) -> String {
        let components = self.get_components();
        let mut result_string = String::new();

        for component in components {
            result_string.push_str(&self.to_gml_with_slice(&component, &[]));
            result_string.push('\n');
        }
        result_string
    }

    fn to_gml_with_slice(&self, filter: &[usize], outer_filter: &[usize]) -> String {
        let mut bonds = String::new();
        let mut atoms = String::new();
        let mut result_string = String::new();
        let atom_bonds = self.atom_bonds();

        for &index in filter.iter() {
            let mut charge = match self.get_atom_charge(index) {
                charge @ 1..=i8::MAX => format!("{}+", charge),
                charge @ i8::MIN..=-1 => format!("{}-", charge.abs()),
                0 => String::new(),
            };
            if self.is_atom_radical(index) {
                charge.push('.');
            }
            let atomic_number = self.get_atomic_number(index);
            let atomic_symbol = atomic_number.atomic_symbol().unwrap_or("Unknown");
            atoms.push_str(&format!(
                "node [id {index} label \"{}{}\"]\n",
                atomic_symbol, charge
            ));
            for &bond in atom_bonds[index].iter() {
                if index < bond.target()
                    && (filter.contains(&bond.target()) || outer_filter.contains(&bond.target()))
                {
                    let label = bond.bond_order().to_string();
                    bonds.push_str(&format!(
                        "edge [source {index} target {} label \"{}\"]\n",
                        bond.target(),
                        label
                    ));
                }
            }
        }
        result_string.push_str("graph [");
        if !atoms.is_empty() {
            result_string.push('\n');
            result_string.push_str(&atoms);
            result_string.push('\n');
        }
        if !bonds.is_empty() {
            if atoms.is_empty() {
                result_string.push('\n');
            }
            result_string.push_str(&bonds);
            result_string.push('\n');
        }
        result_string.push(']');
        result_string
    }
}

/// Represents a node in the GML graph
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier for the node
    pub id: usize,
    /// Chemical element label (e.g., "C", "O", "H", etc.)
    pub label: String,
}

/// Represents an edge in the GML graph
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source node ID
    pub source: usize,
    /// Target node ID
    pub target: usize,
    /// Bond type label (e.g., "-" for single bond)
    pub label: String,
}

/// Represents a section of the reaction rule (left, context, or right)
#[derive(Debug, Clone, Default)]
pub struct GraphSection {
    /// List of nodes in this section
    pub nodes: Vec<Node>,
    /// List of edges in this section
    pub edges: Vec<Edge>,
}

// /// Represents a complete reaction rule in GML format
// #[derive(Debug, Clone)]
// pub struct ReactionRule {
//     /// Unique identifier for the rule
//     pub rule_id: String,
//     /// Left-hand side of the reaction (reactants)
//     pub left: GraphSection,
//     /// Context of the reaction (unchanged parts)
//     pub context: GraphSection,
//     /// Right-hand side of the reaction (products)
//     pub right: GraphSection,
// }

// /// Represents which section of the rule we're currently parsing
// #[derive(Debug, Clone, Copy)]
// enum SectionType {
//     Left,
//     Context,
//     Right,
// }

// impl ReactionRule {
//     /// Creates a new empty reaction rule
//     pub fn new(rule_id: String) -> Self {
//         Self {
//             rule_id,
//             left: GraphSection::default(),
//             context: GraphSection::default(),
//             right: GraphSection::default(),
//         }
//     }

//     /// Parses a GML string into a ReactionRule
//     /// 
//     /// # Examples
//     /// 
//     /// ```
//     /// let gml = r#"
//     /// rule [
//     ///    ruleID "example"
//     ///    left [
//     ///       edge [ source 1 target 2 label "-" ]
//     ///    ]
//     ///    context [
//     ///       node [ id 1 label "O" ]
//     ///       node [ id 2 label "H" ]
//     ///    ]
//     ///    right [
//     ///       edge [ source 2 target 1 label "-" ]
//     ///    ]
//     /// ]
//     /// "#;
//     /// let rule = ReactionRule::from_str(gml).unwrap();
//     /// assert_eq!(rule.rule_id, "example");
//     /// ```
//     pub fn from_str(s: &str) -> Result<Self, String> {
//         let mut lines = s.lines()
//             .map(|l| {
//                 let trimmed = l.trim();
//                 trimmed
//             })
//             .filter(|l| !l.is_empty());
        
//         // Parse the rule header
//         let first_line = lines.next();
//         if first_line != Some("rule [") {
//             return Err("Expected 'rule [' at start".to_string());
//         }

//         let mut rule: Option<ReactionRule> = None;
//         let mut section_type = None;

//         while let Some(line) = lines.next() {
//             match line {
//                 "left [" => {
//                     if rule.is_none() {
//                         return Err("Rule ID must be specified before sections".to_string());
//                     }
//                     section_type = Some(SectionType::Left);
//                 }
//                 "context [" => {
//                     if rule.is_none() {
//                         return Err("Rule ID must be specified before sections".to_string());
//                     }
//                     section_type = Some(SectionType::Context);
//                 }
//                 "right [" => {
//                     if rule.is_none() {
//                         return Err("Rule ID must be specified before sections".to_string());
//                     }
//                     section_type = Some(SectionType::Right);
//                 }
//                 "]" => {
//                     if section_type.is_some() {
//                         section_type = None;
//                     } else {
//                         break; // End of rule
//                     }
//                 }
//                 _ => {
//                     if line.starts_with("ruleID") {
//                         let id = line.split_whitespace().nth(1)
//                             .ok_or("Missing rule ID")?
//                             .trim_matches('"')
//                             .to_string();
//                         rule = Some(ReactionRule::new(id));
//                     } else if line.starts_with("node [") {
//                         let mut node = Node { id: 0, label: String::new() };
                        
//                         // Check if it's a single-line node definition
//                         if line.ends_with("]") {
//                             let content = line.trim_start_matches("node [").trim_end_matches(']');
//                             let mut parts = content.split_whitespace();
//                             while let Some(key) = parts.next() {
//                                 match key {
//                                     "id" => if let Some(val) = parts.next() {
//                                         node.id = val.parse().map_err(|_| "Invalid node ID")?;
//                                     },
//                                     "label" => if let Some(val) = parts.next() {
//                                         node.label = val.trim_matches('"').to_string();
//                                     },
//                                     _ => {}
//                                 }
//                             }
//                         } else {
//                             // Multi-line node definition
//                             while let Some(node_line) = lines.next() {
//                                 let node_line = node_line.trim();
//                                 if node_line == "]" { 
//                                     break; 
//                                 }
//                                 let parts: Vec<&str> = node_line.split_whitespace().collect();
//                                 if parts.len() < 2 { continue; }
//                                 match parts[0] {
//                                     "id" => {
//                                         node.id = parts[1].parse().map_err(|_| "Invalid node ID")?;
//                                     },
//                                     "label" => {
//                                         node.label = parts[1].trim_matches('"').to_string();
//                                     },
//                                     _ => {}
//                                 }
//                             }
//                         }
                        
//                         if let (Some(section), Some(rule)) = (section_type, rule.as_mut()) {
//                             match section {
//                                 SectionType::Left => rule.left.nodes.push(node),
//                                 SectionType::Context => rule.context.nodes.push(node),
//                                 SectionType::Right => rule.right.nodes.push(node),
//                             }
//                         }
//                     } else if line.starts_with("edge [") {
//                         let mut edge = Edge {
//                             source: 0,
//                             target: 0,
//                             label: String::new(),
//                         };
                        
//                         // Check if it's a single-line edge definition
//                         if line.ends_with("]") {
//                             let content = line.trim_start_matches("edge [").trim_end_matches(']');
//                             let mut parts = content.split_whitespace();
//                             while let Some(key) = parts.next() {
//                                 match key {
//                                     "source" => if let Some(val) = parts.next() {
//                                         edge.source = val.parse().map_err(|_| "Invalid source ID")?;
//                                     },
//                                     "target" => if let Some(val) = parts.next() {
//                                         edge.target = val.parse().map_err(|_| "Invalid target ID")?;
//                                     },
//                                     "label" => if let Some(val) = parts.next() {
//                                         edge.label = val.trim_matches('"').to_string();
//                                     },
//                                     _ => {}
//                                 }
//                             }
//                         } else {
//                             // Multi-line edge definition
//                             while let Some(edge_line) = lines.next() {
//                                 let edge_line = edge_line.trim();
//                                 if edge_line == "]" { 
//                                     break; 
//                                 }
//                                 let parts: Vec<&str> = edge_line.split_whitespace().collect();
//                                 if parts.len() < 2 { continue; }
//                                 match parts[0] {
//                                     "source" => {
//                                         edge.source = parts[1].parse().map_err(|_| "Invalid source ID")?;
//                                     },
//                                     "target" => {
//                                         edge.target = parts[1].parse().map_err(|_| "Invalid target ID")?;
//                                     },
//                                     "label" => {
//                                         edge.label = parts[1].trim_matches('"').to_string();
//                                     },
//                                     _ => {}
//                                 }
//                             }
//                         }
                        
//                         if let (Some(section), Some(rule)) = (section_type, rule.as_mut()) {
//                             match section {
//                                 SectionType::Left => rule.left.edges.push(edge),
//                                 SectionType::Context => rule.context.edges.push(edge),
//                                 SectionType::Right => rule.right.edges.push(edge),
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//         let result = rule.ok_or("Failed to parse rule".to_string());
//         result
//     }

//     /// Converts the reaction rule into a petgraph Graph for each section
//     pub fn to_graphs(&self) -> (UnGraph<String, String>, UnGraph<String, String>, UnGraph<String, String>) {
//         let create_graph = |section: &GraphSection| {
//             let mut graph = UnGraph::new_undirected();
//             let mut node_map = HashMap::new();
            
//             // Add nodes
//             for node in &section.nodes {
//                 let node_index = graph.add_node(node.label.clone());
//                 node_map.insert(node.id, node_index);
//             }
            
//             // For sections with edges but no nodes, we need to create the nodes
//             for edge in &section.edges {
//                 if !node_map.contains_key(&edge.source) {
//                     let node_index = graph.add_node(String::new());
//                     node_map.insert(edge.source, node_index);
//                 }
//                 if !node_map.contains_key(&edge.target) {
//                     let node_index = graph.add_node(String::new());
//                     node_map.insert(edge.target, node_index);
//                 }
//             }
            
//             // Add edges
//             for edge in &section.edges {
//                 let source = node_map[&edge.source];
//                 let target = node_map[&edge.target];
//                 graph.add_edge(source, target, edge.label.clone());
//             }
            
//             graph
//         };

//         (
//             create_graph(&self.left),
//             create_graph(&self.context),
//             create_graph(&self.right),
//         )
//     }

//     /// Returns a canonicalized version of the reaction rule
//     pub fn to_canonical(&self) -> ReactionRule {
        
//         // Create a new rule with the same ID
//         let mut rule = ReactionRule::new(self.rule_id.clone());
        
//         // First convert to a canonical graph structure using nauty-pet
//         let graph = self.to_reaction_graph();
//         let mut colored_graph = UnGraph::new_undirected();
//         let mut color_map = HashMap::new();
//         let mut node_indices = HashMap::new();
//         let mut original_indices = HashMap::new();
        
//         // Add nodes with colors based on their labels (ignoring the transformation markers)
//         for (i, node_idx) in graph.node_indices().enumerate() {
//             let label = &graph[node_idx];
            
//             // For coloring, we use just the base element part (ignoring >, + etc.)
//             let base_label = if label.contains('>') {
//                 label.split('>').next().unwrap_or("")
//             } else {
//                 label
//             };
            
//             let color = color_map.len();
//             color_map.entry(base_label.to_string()).or_insert(color);
//             let new_idx = colored_graph.add_node(color_map[base_label]);
//             node_indices.insert(i, new_idx);
//             original_indices.insert(new_idx.index(), node_idx);
//         }
        
//         // Add edges (without labels for canonicalization)
//         for edge in graph.edge_references() {
//             let source_idx = edge.source().index();
//             let target_idx = edge.target().index();
//             let source = node_indices[&source_idx];
//             let target = node_indices[&target_idx];
//             colored_graph.add_edge(source, target, ());
//         }
        
//         // Get the canonicalized graph
//         let canon_graph = colored_graph.into_canon();
        
//         // Create a mapping of new node IDs
//         let mut new_node_ids = HashMap::new();
        
//         // Process nodes in canonical order
//         for (i, node_idx) in canon_graph.node_indices().enumerate() {
//             let original_idx = original_indices[&node_idx.index()];
//             let label = &graph[original_idx];
//             new_node_ids.insert(original_idx.index(), i);
            
//             // Parse the node label to determine which section it belongs to
//             if label.contains('>') {
//                 let parts: Vec<&str> = label.split('>').collect();
//                 match (parts.get(0), parts.get(1)) {
//                     (Some(&""), Some(right)) => {
//                         // Node only in right section (new node)
//                         rule.right.nodes.push(Node {
//                             id: i,
//                             label: right.to_string(),
//                         });
//                     }
//                     (Some(left), Some(&"")) => {
//                         // Node only in left section (deleted node)
//                         rule.left.nodes.push(Node {
//                             id: i,
//                             label: left.to_string(),
//                         });
//                     }
//                     (Some(left), Some(right)) => {
//                         // Node changes from left to right
//                         rule.left.nodes.push(Node {
//                             id: i,
//                             label: left.to_string(),
//                         });
//                         rule.right.nodes.push(Node {
//                             id: i,
//                             label: right.to_string(),
//                         });
//                     }
//                     _ => {}
//                 }
//             } else {
//                 // This is an unchanged node, goes to context
//                 rule.context.nodes.push(Node {
//                     id: i,
//                     label: label.clone(),
//                 });
//             }
//         }
        
//         // Now process edges in the canonical graph
//         for edge in graph.edge_references() {
//             let source_idx = edge.source().index();
//             let target_idx = edge.target().index();
//             let source = new_node_ids[&source_idx];
//             let target = new_node_ids[&target_idx];
//             let label = &graph[edge.id()];
            
//             // Parse the edge label to determine which section it belongs to
//             if label.contains('>') {
//                 let parts: Vec<&str> = label.split('>').collect();
//                 match (parts.get(0), parts.get(1)) {
//                     (Some(&""), Some(right)) => {
//                         // Edge only in right section (new bond)
//                         rule.right.edges.push(Edge {
//                             source,
//                             target,
//                             label: right.to_string(),
//                         });
//                     }
//                     (Some(left), Some(&"")) => {
//                         // Edge only in left section (deleted bond)
//                         rule.left.edges.push(Edge {
//                             source,
//                             target,
//                             label: left.to_string(),
//                         });
//                     }
//                     (Some(left), Some(right)) => {
//                         // Edge changes from left to right
//                         rule.left.edges.push(Edge {
//                             source,
//                             target,
//                             label: left.to_string(),
//                         });
//                         rule.right.edges.push(Edge {
//                             source,
//                             target,
//                             label: right.to_string(),
//                         });
//                     }
//                     _ => {}
//                 }
//             } else {
//                 // This is an unchanged edge, goes to context
//                 rule.context.edges.push(Edge {
//                     source,
//                     target,
//                     label: label.clone(),
//                 });
//             }
//         }
        
//         rule
//     }

//     /// Converts the reaction rule into a combined graph representation
//     /// where nodes and edges that change have labels in the format "LabelLeft>LabelRight"
//     pub fn to_reaction_graph(&self) -> UnGraph<String, String> {
        
//         let mut graph = UnGraph::new_undirected();
//         let mut node_indices = HashMap::new();
        
//         // First, add all nodes from context
//         for node in &self.context.nodes {
//             let node_index = graph.add_node(node.label.clone());
//             node_indices.insert(node.id, node_index);
//         }

//         // Add nodes that appear in both left and right but with different labels
//         // These are transformations like C -> C+
//         for left_node in &self.left.nodes {
//             if let Some(right_node) = self.right.nodes.iter().find(|n| n.id == left_node.id) {
//                 if left_node.label != right_node.label {
//                     // This is a transformation
//                     let label = format!("{}>{}",left_node.label, right_node.label);
//                     if !node_indices.contains_key(&left_node.id) {
//                         let node_index = graph.add_node(label);
//                         node_indices.insert(left_node.id, node_index);
//                     } else {
//                         // Node exists in context but should be updated
//                         let idx = node_indices[&left_node.id];
//                         graph[idx] = label;
//                     }
//                 } else if !node_indices.contains_key(&left_node.id) {
//                     // Node is in both but doesn't change - should be in context
//                     let node_index = graph.add_node(left_node.label.clone());
//                     node_indices.insert(left_node.id, node_index);
//                 }
//             } else if !node_indices.contains_key(&left_node.id) {
//                 // Node only in left (deleted)
//                 let node_index = graph.add_node(format!("{}>", left_node.label));
//                 node_indices.insert(left_node.id, node_index);
//             }
//         }

//         // Add nodes only in right (new nodes)
//         for right_node in &self.right.nodes {
//             if !self.left.nodes.iter().any(|n| n.id == right_node.id) 
//                && !node_indices.contains_key(&right_node.id) {
//                 let node_index = graph.add_node(format!(">{}", right_node.label));
//                 node_indices.insert(right_node.id, node_index);
//             }
//         }

//         // Add edges from context
//         for edge in &self.context.edges {
//             if let (Some(&source), Some(&target)) = (node_indices.get(&edge.source), node_indices.get(&edge.target)) {
//                 graph.add_edge(source, target, edge.label.clone());
//             }
//         }

//         // Add edges from left and right
//         // First find edges that appear in both but with different labels
//         for left_edge in &self.left.edges {
//             if let Some(right_edge) = self.right.edges.iter()
//                 .find(|e| e.source == left_edge.source && e.target == left_edge.target) {
                
//                 if left_edge.label != right_edge.label {
//                     // This is a transformation edge
//                     if let (Some(&source), Some(&target)) = (node_indices.get(&left_edge.source), node_indices.get(&left_edge.target)) {
//                         let label = format!("{}>{}", left_edge.label, right_edge.label);
//                         graph.add_edge(source, target, label);
//                     }
//                 }
//             } else {
//                 // Edge only in left (deleted)
//                 if let (Some(&source), Some(&target)) = (node_indices.get(&left_edge.source), node_indices.get(&left_edge.target)) {
//                     let label = format!("{}>", left_edge.label);
//                     graph.add_edge(source, target, label);
//                 }
//             }
//         }

//         // Add edges only in right (new edges)
//         for right_edge in &self.right.edges {
//             if !self.left.edges.iter().any(|e| e.source == right_edge.source && e.target == right_edge.target) {
//                 if let (Some(&source), Some(&target)) = (node_indices.get(&right_edge.source), node_indices.get(&right_edge.target)) {
//                     let label = format!(">{}", right_edge.label);
//                     graph.add_edge(source, target, label);
//                 }
//             }
//         }

//         graph
//     }

//     /// Writes the reaction graph to a GML file
//     pub fn write_reaction_graph(&self, path: &str) -> std::io::Result<()> {
//         let graph = self.to_reaction_graph();
//         let mut output = String::from("graph [\n");

//         // Write nodes
//         for node_idx in graph.node_indices() {
//             let label = &graph[node_idx];
//             output.push_str(&format!("    node [ id {} label \"{}\" ]\n", 
//                 node_idx.index(), label));
//         }

//         // Write edges
//         for edge in graph.edge_references() {
//             output.push_str(&format!("    edge [ source {} target {} label \"{}\" ]\n",
//                 edge.source().index(), edge.target().index(), &graph[edge.id()]));
//         }

//         output.push_str("]\n");
//         std::fs::write(path, output)
//     }

//     /// Writes the canonical reaction rule to a GML file
//     pub fn write_canonical(&self, path: &str) -> std::io::Result<()> {
//         let canonical = self.to_canonical();
//         let mut output = String::from("rule [\n");
//         output.push_str(&format!("    ruleID \"{}\"\n", canonical.rule_id));
        
//         // Write left section
//         output.push_str("    left [\n");
//         for edge in &canonical.left.edges {
//             output.push_str(&format!("        edge [ source {} target {} label \"{}\" ]\n",
//                 edge.source, edge.target, edge.label));
//         }
//         output.push_str("    ]\n");
        
//         // Write context section
//         output.push_str("    context [\n");
//         for node in &canonical.context.nodes {
//             output.push_str(&format!("        node [ id {} label \"{}\" ]\n",
//                 node.id, node.label));
//         }
//         for edge in &canonical.context.edges {
//             output.push_str(&format!("        edge [ source {} target {} label \"{}\" ]\n",
//                 edge.source, edge.target, edge.label));
//         }
//         output.push_str("    ]\n");
        
//         // Write right section
//         output.push_str("    right [\n");
//         for edge in &canonical.right.edges {
//             output.push_str(&format!("        edge [ source {} target {} label \"{}\" ]\n",
//                 edge.source, edge.target, edge.label));
//         }
//         output.push_str("    ]\n");
//         output.push_str("]\n");
        
//         std::fs::write(path, output)
//     }
// }

// impl FromStr for ReactionRule {
//     type Err = String;

//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         ReactionRule::from_str(s)
//     }
// } 

// #[cfg(test)]
// mod rule_gml_tests {
//     use super::*;

//     #[test]
//     fn test_reaction_rule_from_str() {
//         let gml = r#"rule [
//     ruleID "rule"
//     left [
//        edge [ source 1 target 2 label "-" ]
//        edge [ source 3 target 4 label "-" ]
//     ]
//     context [
//        node [ id 1 label "O" ]
//        node [ id 2 label "Si" ]
//        node [ id 3 label "H" ]
//        node [ id 4 label "O" ]
//        node [ id 5 label "C" ]
//     ]
//     right [
//        edge [ source 1 target 3 label "-" ]
//        edge [ source 2 target 4 label "-" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
        
//         // Check rule ID
//         assert_eq!(rule.rule_id, "rule");
        
//         // Check left section
//         assert_eq!(rule.left.edges.len(), 2);
//         assert_eq!(rule.left.edges[0].source, 1);
//         assert_eq!(rule.left.edges[0].target, 2);
//         assert_eq!(rule.left.edges[0].label, "-");
        
//         // Check context section
//         assert_eq!(rule.context.nodes.len(), 5);
//         assert_eq!(rule.context.nodes[0].id, 1);
//         assert_eq!(rule.context.nodes[0].label, "O");
//         assert_eq!(rule.context.nodes[1].label, "Si");
        
//         // Check right section
//         assert_eq!(rule.right.edges.len(), 2);
//         assert_eq!(rule.right.edges[0].source, 1);
//         assert_eq!(rule.right.edges[0].target, 3);
//         assert_eq!(rule.right.edges[0].label, "-");

//         // Check graph conversion
//         let (left, context, right) = rule.to_graphs();
//         assert_eq!(left.edge_count(), 2);
//         assert_eq!(context.node_count(), 5);
//         assert_eq!(right.edge_count(), 2);
//     }

//     #[test]
//     fn test_empty_sections() {
//         let gml = r#"rule [
//     ruleID "empty_test"
//     left [
//     ]
//     context [
//     ]
//     right [
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
//         assert_eq!(rule.rule_id, "empty_test");
//         assert!(rule.left.nodes.is_empty());
//         assert!(rule.left.edges.is_empty());
//         assert!(rule.context.nodes.is_empty());
//         assert!(rule.context.edges.is_empty());
//         assert!(rule.right.nodes.is_empty());
//         assert!(rule.right.edges.is_empty());
//     }

//     #[test]
//     fn test_multiline_format() {
//         let gml = r#"rule [
//     ruleID "multiline"
//     left [
//         edge [
//             source 1
//             target 2
//             label "-"
//         ]
//     ]
//     context [
//         node [
//             id 1
//             label "C"
//         ]
//         node [
//             id 2
//             label "O"
//         ]
//     ]
//     right [
//         edge [
//             source 2
//             target 1
//             label "="
//         ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
//         assert_eq!(rule.rule_id, "multiline");
        
//         // Check left section
//         assert_eq!(rule.left.edges.len(), 1);
//         assert_eq!(rule.left.edges[0].source, 1);
//         assert_eq!(rule.left.edges[0].target, 2);
//         assert_eq!(rule.left.edges[0].label, "-");
        
//         // Check context section
//         assert_eq!(rule.context.nodes.len(), 2);
//         assert_eq!(rule.context.nodes[0].label, "C");
//         assert_eq!(rule.context.nodes[1].label, "O");
        
//         // Check right section
//         assert_eq!(rule.right.edges.len(), 1);
//         assert_eq!(rule.right.edges[0].source, 2);
//         assert_eq!(rule.right.edges[0].target, 1);
//         assert_eq!(rule.right.edges[0].label, "=");
//     }

//     #[test]
//     fn test_mixed_format() {
//         let gml = r#"rule [
//     ruleID "mixed"
//     left [
//         edge [ source 1 target 2 label "-" ]
//         edge [
//             source 2
//             target 3
//             label "="
//         ]
//     ]
//     context [
//         node [ id 1 label "C" ]
//         node [
//             id 2
//             label "O"
//         ]
//         node [ id 3 label "N" ]
//         edge [ source 1 target 3 label "-" ]
//     ]
//     right [
//         edge [ source 1 target 2 label "=" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
//         assert_eq!(rule.rule_id, "mixed");
        
//         // Check left section
//         assert_eq!(rule.left.edges.len(), 2);
//         assert_eq!(rule.left.edges[0].label, "-");
//         assert_eq!(rule.left.edges[1].label, "=");
        
//         // Check context section
//         assert_eq!(rule.context.nodes.len(), 3);
//         assert_eq!(rule.context.nodes[0].label, "C");
//         assert_eq!(rule.context.nodes[1].label, "O");
//         assert_eq!(rule.context.nodes[2].label, "N");
//         assert_eq!(rule.context.edges.len(), 1);
        
//         // Check right section
//         assert_eq!(rule.right.edges.len(), 1);
//         assert_eq!(rule.right.edges[0].source, 1);
//         assert_eq!(rule.right.edges[0].target, 2);
//         assert_eq!(rule.right.edges[0].label, "=");

//         // Check graph conversion
//         let (left, context, right) = rule.to_graphs();
//         assert_eq!(left.edge_count(), 2);
//         assert_eq!(left.node_count(), 3);
//         assert_eq!(context.node_count(), 3);
//         assert_eq!(context.edge_count(), 1);
//         assert_eq!(right.edge_count(), 1);
//         assert_eq!(right.node_count(), 2);
//     }

//     #[test]
//     fn test_error_cases() {
//         // Missing rule header
//         assert!(ReactionRule::from_str("left []").is_err());
        
//         // Missing ruleID
//         assert!(ReactionRule::from_str("rule [\nleft []\n]").is_err());
        
//         // Invalid node ID
//         let invalid_node = r#"rule [
//     ruleID "invalid"
//     context [
//         node [ id abc label "C" ]
//     ]
// ]"#;
//         assert!(ReactionRule::from_str(invalid_node).is_err());
        
//         // Invalid edge source
//         let invalid_edge = r#"rule [
//     ruleID "invalid"
//     left [
//         edge [ source xyz target 2 label "-" ]
//     ]
// ]"#;
//         assert!(ReactionRule::from_str(invalid_edge).is_err());
//     }

//     #[test]
//     fn test_large_reaction_rule() {
//         let gml = r#"rule [
//     ruleID "rule"
//     left [
//         edge [ source 1 target 2 label "-" ]
//         edge [ source 3 target 4 label "-" ]
//     ]
//     context [
//         node [ id 1 label "O" ]
//         node [ id 2 label "Si" ]
//         node [ id 3 label "H" ]
//         node [ id 4 label "O" ]
//         node [ id 5 label "C" ]
//         node [ id 6 label "C" ]
//         node [ id 9 label "C" ]
//         node [ id 10 label "C" ]
//         node [ id 11 label "C" ]
//         node [ id 12 label "C" ]
//         node [ id 13 label "C" ]
//         node [ id 14 label "C" ]
//         node [ id 15 label "C" ]
//         node [ id 16 label "F" ]
//         node [ id 17 label "C" ]
//         node [ id 18 label "C" ]
//         node [ id 19 label "N" ]
//         node [ id 20 label "C" ]
//         node [ id 21 label "C" ]
//         node [ id 22 label "C" ]
//         node [ id 23 label "C" ]
//         node [ id 24 label "C" ]
//         node [ id 25 label "N" ]
//         node [ id 26 label "C" ]
//         node [ id 27 label "O" ]
//         node [ id 28 label "O" ]
//         node [ id 29 label "C" ]
//         node [ id 30 label "C" ]
//         node [ id 31 label "C" ]
//         node [ id 32 label "C" ]
//         node [ id 33 label "C" ]
//         node [ id 34 label "O" ]
//         node [ id 35 label "O" ]
//         node [ id 36 label "C" ]
//         node [ id 37 label "C" ]
//         node [ id 38 label "C" ]
//         node [ id 39 label "C" ]
//         node [ id 40 label "N" ]
//         node [ id 41 label "C" ]
//         node [ id 42 label "C" ]
//         node [ id 43 label "C" ]
//         node [ id 44 label "C" ]
//         node [ id 45 label "C" ]
//         node [ id 46 label "N" ]
//         node [ id 47 label "C" ]
//         node [ id 48 label "C" ]
//         node [ id 49 label "C" ]
//         node [ id 50 label "C" ]
//         node [ id 51 label "C" ]
//         node [ id 52 label "C" ]
//         node [ id 53 label "C" ]
//         node [ id 54 label "C" ]
//         node [ id 55 label "N" ]
//         node [ id 56 label "C" ]
//         node [ id 57 label "C" ]
//         node [ id 58 label "C" ]
//         node [ id 59 label "C" ]
//         node [ id 60 label "C" ]
//         node [ id 61 label "C" ]
//         node [ id 62 label "C" ]
//         node [ id 63 label "C" ]
//         node [ id 64 label "C" ]
//         node [ id 65 label "C" ]
//         node [ id 66 label "C" ]
//         node [ id 67 label "C" ]
//         node [ id 68 label "C" ]
//         node [ id 69 label "C" ]
//         node [ id 70 label "C" ]
//         node [ id 71 label "C" ]
//         node [ id 72 label "C" ]
//         node [ id 73 label "C" ]
//         node [ id 74 label "C" ]
//         node [ id 75 label "C" ]
//     ]
//     right [
//         edge [ source 1 target 3 label "-" ]
//         edge [ source 2 target 4 label "-" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
        
//         // Check rule ID
//         assert_eq!(rule.rule_id, "rule");
        
//         // Check left section
//         assert_eq!(rule.left.edges.len(), 2);
//         assert_eq!(rule.left.edges[0].source, 1);
//         assert_eq!(rule.left.edges[0].target, 2);
//         assert_eq!(rule.left.edges[0].label, "-");
        
//         // Check context section
//         assert_eq!(rule.context.nodes.len(), 73, "Should have all 73 nodes");
//         assert_eq!(rule.context.nodes[0].id, 1);
//         assert_eq!(rule.context.nodes[0].label, "O");
//         assert_eq!(rule.context.nodes[1].label, "Si");
        
//         // Check right section
//         assert_eq!(rule.right.edges.len(), 2);
//         assert_eq!(rule.right.edges[0].source, 1);
//         assert_eq!(rule.right.edges[0].target, 3);
//         assert_eq!(rule.right.edges[0].label, "-");

//         // Check canonicalization
//         let canonical = rule.to_canonical();
//         println!("Canonical Rule: {:?}", canonical);
//         canonical.write_canonical("test_reaction.gml").unwrap();
        
//         // Read and verify the canonical file
//         let content = std::fs::read_to_string("test_reaction.gml").unwrap();
//         println!("Content: {:?}", content);
//         assert!(content.contains("rule ["));
//         assert!(content.contains("ruleID"));
//         assert!(content.contains("left ["));
//         assert!(content.contains("context ["));
//         assert!(content.contains("right ["));
        
//         // Clean up
//         std::fs::remove_file("test_reaction.gml").unwrap();
//     }
// }

// #[cfg(test)]
// mod reaction_graph_tests {
//     use super::*;

//     #[test]
//     fn test_reaction_graph_creation() {
//         let gml = r#"rule [
//     ruleID "test_reaction"
//     left [
//         edge [ source 1 target 2 label "-" ]
//     ]
//     context [
//         node [ id 1 label "C" ]
//         node [ id 2 label "O" ]
//         node [ id 3 label "C" ]
//         edge [ source 2 target 3 label "-" ]
//     ]
//     right [
//         edge [ source 1 target 2 label "=" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
//         let graph = rule.to_reaction_graph();

//         println!("Graph: {:?}", graph);
//         // Check nodes
//         assert_eq!(graph.node_count(), 3);
        
//         // Find the edge that changes
//         let mut found_changing_edge = false;
//         for edge in graph.edge_references() {
//             let label = &graph[edge.id()];
//             if label == "->=" {
//                 found_changing_edge = true;
//                 assert_eq!(graph[edge.source()], "C");
//                 assert_eq!(graph[edge.target()], "O");
//             }
//         }

//         assert!(found_changing_edge, "Should find the edge that changes from - to =");
//     }

//     #[test]
//     fn test_reaction_graph_file_writing() {
//         let gml = r#"rule [
//     ruleID "test_write"
//     left [
//         edge [ source 1 target 2 label "-" ]
//     ]
//     context [
//         node [ id 1 label "C" ]
//         node [ id 2 label "O" ]
//     ]
//     right [
//         edge [ source 1 target 2 label "=" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
//         rule.write_canonical("test_reaction.gml").unwrap();

//         // Read and verify the file
//         // let content = std::fs::read_to_string("test_reaction.gml").unwrap();
//         // assert!(content.contains("rule ["));
//         // assert!(content.contains("node [ id"));
//         // assert!(content.contains("edge [ source"));
//         // assert!(content.contains("->="));  // Check for the transformed edge label
        
//         // Clean up
//         std::fs::remove_file("test_reaction.gml").unwrap();
//     }
// }

// #[cfg(test)]
// mod node_transformation_tests {
//     use super::*;

//     #[test]
//     fn test_node_charge_transformation() {
//         let gml = r#"rule [
//     ruleID "charge_change"
//     left [
//         node [ id 1 label "C" ]
//         edge [ source 1 target 2 label "-" ]
//     ]
//     context [
//         node [ id 2 label "O" ]
//     ]
//     right [
//         node [ id 1 label "C+" ]
//         edge [ source 1 target 2 label "-" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
        
//         // Check that node 1 appears in both left and right with different labels
//         assert_eq!(rule.left.nodes[0].label, "C");
//         assert_eq!(rule.right.nodes[0].label, "C+");
        
//         // Check that node 2 remains unchanged in context
//         assert_eq!(rule.context.nodes[0].label, "O");
        
//         // Test canonicalization preserves the transformation
//         let canonical = rule.to_canonical();
//         assert!(canonical.left.nodes.iter().any(|n| n.label == "C"));
//         assert!(canonical.right.nodes.iter().any(|n| n.label == "C+"));
//     }

//     #[test]
//     fn test_node_radical_transformation() {
//         let gml = r#"rule [
//     ruleID "radical_formation"
//     left [
//         node [ id 1 label "C" ]
//         node [ id 2 label "H" ]
//         edge [ source 1 target 2 label "-" ]
//     ]
//     context [
//         node [ id 3 label "O" ]
//         edge [ source 1 target 3 label "-" ]
//     ]
//     right [
//         node [ id 1 label "C." ]
//         edge [ source 1 target 3 label "-" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
        
//         // Check node transformations
//         assert_eq!(rule.left.nodes[0].label, "C");
//         assert_eq!(rule.right.nodes[0].label, "C.");
        
//         // Check that H is only in left (gets removed)
//         assert!(rule.left.nodes.iter().any(|n| n.label == "H"));
//         assert!(!rule.right.nodes.iter().any(|n| n.label == "H"));
        
//         // Test canonicalization
//         let canonical = rule.to_canonical();
//         assert!(canonical.left.nodes.iter().any(|n| n.label == "C"));
//         assert!(canonical.right.nodes.iter().any(|n| n.label == "C."));
//         assert!(canonical.left.nodes.iter().any(|n| n.label == "H"));
//     }

//     #[test]
//     fn test_node_appearance_and_disappearance() {
//         let gml = r#"rule [
//     ruleID "node_changes"
//     left [
//         node [ id 1 label "C" ]
//         edge [ source 1 target 2 label "-" ]
//     ]
//     context [
//         node [ id 2 label "O" ]
//     ]
//     right [
//         node [ id 3 label "N" ]
//         edge [ source 2 target 3 label "-" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
        
//         // Check node appearances/disappearances
//         assert!(rule.left.nodes.iter().any(|n| n.label == "C"));
//         assert!(!rule.right.nodes.iter().any(|n| n.label == "C")); // C disappears
//         assert!(!rule.left.nodes.iter().any(|n| n.label == "N"));
//         assert!(rule.right.nodes.iter().any(|n| n.label == "N")); // N appears
        
//         // Test canonicalization
//         let canonical = rule.to_canonical();
//         assert!(canonical.left.nodes.iter().any(|n| n.label == "C"));
//         assert!(canonical.right.nodes.iter().any(|n| n.label == "N"));
//     }

//     #[test]
//     fn test_complex_transformation() {
//         let gml = r#"rule [
//     ruleID "complex_change"
//     left [
//         node [ id 1 label "C" ]
//         node [ id 2 label "O" ]
//         node [ id 3 label "H" ]
//         edge [ source 1 target 2 label "-" ]
//         edge [ source 2 target 3 label "-" ]
//     ]
//     context [
//         node [ id 4 label "N" ]
//         edge [ source 1 target 4 label "-" ]
//     ]
//     right [
//         node [ id 1 label "C+" ]
//         node [ id 2 label "O-" ]
//         edge [ source 1 target 2 label "=" ]
//         edge [ source 1 target 4 label "-" ]
//     ]
// ]"#;
//         let rule = ReactionRule::from_str(gml).unwrap();
        
//         // Check node transformations
//         assert!(rule.left.nodes.iter().any(|n| n.label == "C"));
//         assert!(rule.right.nodes.iter().any(|n| n.label == "C+"));
//         assert!(rule.left.nodes.iter().any(|n| n.label == "O"));
//         assert!(rule.right.nodes.iter().any(|n| n.label == "O-"));
        
//         // Check edge transformations
//         assert!(rule.left.edges.iter().any(|e| e.label == "-" && 
//             e.source == 1 && e.target == 2));
//         assert!(rule.right.edges.iter().any(|e| e.label == "=" && 
//             e.source == 1 && e.target == 2));
        
//         // Check H disappears
//         assert!(rule.left.nodes.iter().any(|n| n.label == "H"));
//         assert!(!rule.right.nodes.iter().any(|n| n.label == "H"));
        
//         // Test canonicalization preserves all transformations
//         let canonical = rule.to_canonical();
//         println!("Canonical Rule: {:?}", canonical);
        
//         // Node transformations preserved
//         assert!(canonical.left.nodes.iter().any(|n| n.label == "C"));
//         assert!(canonical.right.nodes.iter().any(|n| n.label == "C+"));
//         assert!(canonical.left.nodes.iter().any(|n| n.label == "O"));
//         assert!(canonical.right.nodes.iter().any(|n| n.label == "O-"));
        
//         // Edge transformations preserved
//         let has_edge_transform = canonical.left.edges.iter().zip(canonical.right.edges.iter())
//             .any(|(l, r)| l.label == "-" && r.label == "=" && 
//                 l.source == r.source && l.target == r.target);
//         assert!(has_edge_transform);
        
//         // Node deletion preserved
//         assert!(canonical.left.nodes.iter().any(|n| n.label == "H"));
//         assert!(!canonical.right.nodes.iter().any(|n| n.label == "H"));
//     }
// }
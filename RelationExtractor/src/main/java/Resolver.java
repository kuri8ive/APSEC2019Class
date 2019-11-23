import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.FieldAccessExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Resolver {
    private HashMap<String, ArrayList<String>> classExtends;
    private HashMap<String, ArrayList<String>> fieldsInClass;
    private HashMap<String, ArrayList<String>> fieldType;
    private HashMap<String, ArrayList<String>> methodsInClass;
    private HashMap<String, ArrayList<String>> methodCall;
    private HashMap<String, ArrayList<String>> fieldsInMethod;
    private HashMap<String, ArrayList<String>> returnType;

    private HashMap<String, HashMap<String, ArrayList<String>>> relations;

    private HashSet<String> declarationClassSet;
    private HashSet<String> extendedClassSet;
    private HashSet<String> fieldInClassSet;
    private HashSet<String> fieldTypeSet;
    private HashSet<String> methodInClassSet;
    private HashSet<String> fieldInMethodSet;
    private HashSet<String> calleeSet;
    private HashSet<String> returnTypeSet;

    private boolean doNameResolve;
    private int resolveSuccess;
    private int resolveFailed;

    public int getResolveSuccess() {
        return resolveSuccess;
    }

    public int getResolveFailed() {
        return resolveFailed;
    }

    Resolver() {
        classExtends = new HashMap<>();
        methodsInClass = new HashMap<>();
        fieldsInClass = new HashMap<>();
        methodCall = new HashMap<>();
        fieldsInMethod = new HashMap<>();
        returnType = new HashMap<>();
        fieldType = new HashMap<>();

        relations = new HashMap<>();

        doNameResolve = true;
        resolveSuccess = 0;
        resolveFailed = 0;

        declarationClassSet = new HashSet<>();
        extendedClassSet = new HashSet<>();
        fieldInClassSet = new HashSet<>();
        fieldTypeSet = new HashSet<>();
        methodInClassSet = new HashSet<>();
        fieldInMethodSet = new HashSet<>();
        calleeSet = new HashSet<>();
        returnTypeSet = new HashSet<>();
    }

    Resolver(boolean doNameResolve) {
        this();
        this.doNameResolve = doNameResolve;
    }

    public HashMap<String, ArrayList<String>> getClassExtends() {
        return classExtends;
    }

    public HashMap<String, ArrayList<String>> getMethodsInClass() {
        return methodsInClass;
    }

    public HashMap<String, ArrayList<String>> getFieldsInClass() {
        return fieldsInClass;
    }

    public HashMap<String, ArrayList<String>> getMethodCall() {
        return methodCall;
    }

    public HashMap<String, ArrayList<String>> getFieldsInMethod() {
        return fieldsInMethod;
    }

    public HashMap<String, ArrayList<String>> getReturnType() {
        return returnType;
    }

    public HashMap<String, ArrayList<String>> getFieldType() {
        return fieldType;
    }

    public boolean isParam1inParam2(String part, String str) {
        if (str.matches(".*" + part + ".*")) {
            return true;
        } else {
            return false;
        }
    }

    public HashMap<String, Integer> getEachElementNum() {
        HashMap<String, Integer> countResults = new HashMap<>();
        countResults.put("declarationClass", declarationClassSet.size());
        countResults.put("extendedClass", extendedClassSet.size());
        countResults.put("fieldInClass", fieldInClassSet.size());
        countResults.put("fieldType", fieldTypeSet.size());
        countResults.put("methodInClass", methodInClassSet.size());
        countResults.put("fieldInMethod", fieldInMethodSet.size());
        countResults.put("returnType", returnTypeSet.size());
        return countResults;
    }

    public void refreshRelations() {
        classExtends = new HashMap<>();
        methodsInClass = new HashMap<>();
        fieldsInClass = new HashMap<>();
        methodCall = new HashMap<>();
        fieldsInMethod = new HashMap<>();
        returnType = new HashMap<>();
        fieldType = new HashMap<>();
    }

    private void getRelationIfExists(String relationName, HashMap<String, ArrayList<String>> relation) {
        if (!relation.isEmpty())
            relations.put(relationName, relation);
    }

    public HashMap<String, HashMap<String, ArrayList<String>>> getRelations() {
        getRelationIfExists("classExtends", getClassExtends());
        getRelationIfExists("methodInClass", getMethodsInClass());
        getRelationIfExists("fieldInClass", getFieldsInClass());
        getRelationIfExists("methodCall", getMethodCall());
        getRelationIfExists("fieldInMethod", getFieldsInMethod());
        getRelationIfExists("returnType", getReturnType());
        getRelationIfExists("fieldType", getFieldType());
        return relations;
    }

    public void execute(CompilationUnit compilationUnit) {

        compilationUnit.findAll(ClassOrInterfaceDeclaration.class).forEach(classOrInterfaceDeclaration -> {
            if (classOrInterfaceDeclaration.isInterface())
                return;

            String resolvedDeclarationClassName = resolveClass(classOrInterfaceDeclaration);
            declarationClassSet.add(resolvedDeclarationClassName);

            ArrayList<String> tmpArray;

            // 継承クラスの処理
            tmpArray = new ArrayList<>();
            resolveExtendedClass(classOrInterfaceDeclaration, tmpArray, resolvedDeclarationClassName);
            if (!tmpArray.isEmpty())
                classExtends.put(resolvedDeclarationClassName, tmpArray);

            classOrInterfaceDeclaration.findAll(FieldDeclaration.class).forEach(fieldDeclaration -> {
                ArrayList<String> strings = new ArrayList<>();

                String fieldName = resolvedDeclarationClassName + "."
                        + fieldDeclaration.getVariable(0).getNameAsString();
                strings.add(fieldName);
                if (fieldsInClass.containsKey(resolvedDeclarationClassName)) {
                    fieldsInClass.get(resolvedDeclarationClassName).add(fieldName);
                } else {
                    fieldsInClass.put(resolvedDeclarationClassName, strings);
                }
                fieldInClassSet.add(fieldName);

                strings = new ArrayList<>();
                resolveFieldType(fieldDeclaration, strings, resolvedDeclarationClassName);
                fieldType.put(fieldName, strings);

                strings = new ArrayList<>();
                resolveNewInField(fieldDeclaration, strings, resolvedDeclarationClassName);
                if (!strings.isEmpty())
                    methodsInClass.put(resolvedDeclarationClassName, strings);

            });

            classOrInterfaceDeclaration.findAll(MethodDeclaration.class).forEach(methodDeclaration -> {
                ArrayList<String> strings = new ArrayList<>();
                String declarationMethodName = resolvedDeclarationClassName + "." + methodDeclaration.getNameAsString();
                strings.add(declarationMethodName);
                if (methodsInClass.containsKey(resolvedDeclarationClassName)) {
                    methodsInClass.get(resolvedDeclarationClassName).add(declarationMethodName);
                } else {
                    methodsInClass.put(resolvedDeclarationClassName, strings);
                }

                strings = new ArrayList<>();
                resolveMethodCall(methodDeclaration, strings, resolvedDeclarationClassName);
                if (!strings.isEmpty())
                    methodCall.put(declarationMethodName, strings);

                strings = new ArrayList<>();
                resolveNewInMethod(methodDeclaration, strings, resolvedDeclarationClassName);
                if (!strings.isEmpty())
                    methodCall.put(declarationMethodName, strings);

                strings = new ArrayList<>();
                resolveFieldAccess(methodDeclaration, strings, resolvedDeclarationClassName);
                if (!strings.isEmpty())
                    fieldsInMethod.put(declarationMethodName, strings);

                strings = new ArrayList<>();
                resolveReturnType(methodDeclaration, strings, resolvedDeclarationClassName);
                returnType.put(declarationMethodName, strings);
            });
        });
    }

    private String resolveClass(ClassOrInterfaceDeclaration classDeclaration) {
        String name = classDeclaration.getNameAsString();
        if (!doNameResolve)
            return name;
        try {
            name = classDeclaration.resolve().getQualifiedName();
            // System.out.print("\r Resolve Success : " + name);
            resolveSuccess++;
            return name;
        } catch (Throwable t) {
            // System.out.print("\r Resolve Failed : " + name + " @ ");

            resolveFailed++;
            return name;
        }
    }

    private void resolveExtendedClass(ClassOrInterfaceDeclaration classDeclaration, ArrayList<String> arrayList,
            String resolvedDeclarationClassName) {
        ClassOrInterfaceType parent;
        if (classDeclaration.getExtendedTypes().size() != 0) {
            parent = classDeclaration.getExtendedTypes(0);
            try {
                parent = classDeclaration.getExtendedTypes(0).asClassOrInterfaceType();
            } catch (Throwable t) {
                // System.out.println(resolvedDeclarationClassName);
            }
        } else {
            // if (doNameResolve){
            // arrayList.add("java.lang.Object");
            // resolveSuccess++;
            // } else {
            // arrayList.add("Object");
            // }
            return;
        }

        String name = parent.getNameAsString();
        if (doNameResolve) {
            try {
                name = parent.resolve().getQualifiedName();
                // System.out.print("\r classExtends Resolve Success : " + name);
                resolveSuccess++;
            } catch (Throwable t) {
                // System.out.print("\r classExtends Resolve Failed : " + name + " @ ");
                // System.out.println(resolvedDeclarationClassName);
                resolveFailed++;
            }
            arrayList.add(name);
        } else {
            arrayList.add(name);
        }
        extendedClassSet.add(name);
    }

    private String resolveFieldInClass(FieldDeclaration fieldDeclaration, ArrayList<String> arrayList,
            String filePath) {
        String fieldName = fieldDeclaration.getVariable(0).getNameAsString();
        if (!doNameResolve) {
            arrayList.add(fieldName);
            return fieldName;
        }
        try {
            fieldName = fieldDeclaration.resolve().getType().asTypeVariable().qualifiedName() + "." + fieldName;
            fieldName = fieldName;
        } catch (Throwable t) {
            fieldName = fieldName;
        }
        arrayList.add(fieldName);
        return fieldName;
    }

    private void resolveFieldType(FieldDeclaration fieldDeclaration, ArrayList<String> arrayList, String filePath) {
        String name = fieldDeclaration.getVariable(0).getTypeAsString();
        if (doNameResolve) {
            try {
                name = fieldDeclaration.getVariable(0).getType().resolve().describe();// .asClassOrInterfaceType().resolve().describe();
                // System.out.print("\r fieldType Resolve Success : " + name);
                resolveSuccess++;
            } catch (IllegalStateException i) {
                // System.out.print("\r fieldType Resolve Success : " + name);
                resolveSuccess++;
            } catch (UnsupportedOperationException t) {
                if (name.equals("void")) {
                    // System.out.print("\r returnType Resolve Success : " + name);
                    resolveSuccess++;
                } else {
                    // System.out.print("\r fieldType Resolve Failed : " + name + " -> " + "\n");
                    // System.out.print(filePath);
                    // System.out.println(" -> " + t.toString() + "\n");
                    resolveFailed++;
                }
            } catch (Throwable t) {
                // System.out.print("\r fieldType Resolve Failed : " + name + " @ ");
                // System.out.print(filePath);
                // System.out.println(" -> " + t.toString() + "\n");
                resolveFailed++;
            }
            arrayList.add(name);
        } else {
            arrayList.add(name);
        }
        fieldTypeSet.add(name);
    }

    private void resolveNewInField(FieldDeclaration fieldDeclaration, ArrayList<String> arrayList,
            String resolvedDeclarationClassName) {
        if (!fieldDeclaration.isInitializerDeclaration())
            return;
        fieldDeclaration.findAll(ObjectCreationExpr.class).forEach(objectCreationExpr -> {
            String name = objectCreationExpr.getTypeAsString();
            if (!doNameResolve) {
                arrayList.add(name);
            } else {
                try {
                    name = objectCreationExpr.getType().resolve().describe() + "." + name;
                    resolveSuccess++;
                } catch (Exception e) {
                    // System.out.print("\r newInField Resolve Failed, newInField name : " + name);
                    // System.out.println(" in a file of " + resolvedDeclarationClassName + ", class
                    // is " + objectCreationExpr.getTypeAsString() + "\n");
                    resolveFailed++;
                }
                arrayList.add(name);
            }
            methodInClassSet.add(name);
        });
    }

    private String resolveMethodInClass(MethodDeclaration declarationMethod, ArrayList<String> arrayList,
            String filePath) {
        String methodName = declarationMethod.getNameAsString();
        if (!doNameResolve) {
            arrayList.add(methodName);
            return methodName;
        }
        try {
            methodName = declarationMethod.resolve().getQualifiedName();
            methodName = methodName;
        } catch (Throwable t) {
            methodName = methodName;
        }
        arrayList.add(methodName);
        return methodName;
    }

    private void resolveMethodCall(MethodDeclaration methodDeclaration, ArrayList<String> arrayList,
            String resolvedDeclarationClassName) {
        methodDeclaration.findAll(MethodCallExpr.class).forEach(callee -> {
            String name = callee.getNameAsString();

            if (doNameResolve) {
                if (!callee.getScope().isPresent()) {
                    arrayList.add(resolvedDeclarationClassName + "." + name);
                    resolveSuccess++;
                    return;
                }
                try {
                    name = callee.getScope().get().calculateResolvedType().describe() + "." + name;
                    // System.out.print("\r methodCall Resolve Success : " + name);
                    resolveSuccess++;
                } catch (Throwable t) {
                    // System.out.print("\r methodCall Resolve Failed, method name : " + name);
                    // System.out.println(" in a file of " + resolvedDeclarationClassName + ", scope
                    // is " + callee.getScope().get().toString() + "\n");
                    resolveFailed++;
                }
                arrayList.add(name);
            } else {
                arrayList.add(name);
            }
            calleeSet.add(name);
        });
    }

    private void resolveNewInMethod(MethodDeclaration methodDeclaration, ArrayList<String> arrayList,
            String resolvedDeclarationClassName) {
        methodDeclaration.findAll(ObjectCreationExpr.class).forEach(objectCreationExpr -> {
            String name = objectCreationExpr.getTypeAsString();

            if (doNameResolve) {
                try {
                    name = objectCreationExpr.getType().resolve().describe() + "." + name;
                    // System.out.print("\r methodCall Resolve Success : " + name);
                    resolveSuccess++;
                } catch (Throwable t) {
                    // System.out.print("\r newInClass Resolve Failed, newInClass name : " + name);
                    // System.out.println(" in a file of " + resolvedDeclarationClassName + ", class
                    // is " + name + "\n");
                    resolveFailed++;
                }
                arrayList.add(name);
            } else {
                arrayList.add(name);
            }
            calleeSet.add(name);
        });
    }

    private void resolveFieldAccess(MethodDeclaration declarationMethod, ArrayList<String> arrayList, String filePath) {
        declarationMethod.findAll(FieldAccessExpr.class).forEach(accessedField -> {
            if (accessedField.getParentNode().isPresent()) {
                if (accessedField.getParentNode().get() instanceof FieldAccessExpr)
                    return;
            }
            String name = accessedField.getNameAsString();
            if (doNameResolve) {
                try {
                    name = accessedField.getScope().calculateResolvedType().describe() + "." + name;
                    // System.out.print("\r fieldAccess Resolve Success : " + name);
                    resolveSuccess++;
                } catch (Throwable t) {
                    // System.out.print("\r fieldAccess Resolve Failed, field name : " + name);
                    // System.out.println(", in a file of : " + filePath + ", scope is : " +
                    // accessedField.getScope().toString() + "\n");
                    Expression scope = accessedField.getScope();
                    if (scope != null) {
                        if (isParam1inParam2("java", scope.toString())) {
                            name = scope.toString() + "." + name;
                            resolveSuccess++;
                        }
                    } else
                        resolveFailed++;
                }
                arrayList.add(name);
            } else {
                arrayList.add(name);
            }
            fieldInMethodSet.add(name);
        });
    }

    private void resolveReturnType(MethodDeclaration declarationMethod, ArrayList<String> arrayList, String filePath) {
        String name = declarationMethod.getType().asString();
        if (doNameResolve) {
            try {
                name = declarationMethod.getType().resolve().describe();
                // System.out.print("\r returnType Resolve Success : " + name);
                resolveSuccess++;
            } catch (UnsupportedOperationException t) {
                if (isParam1inParam2("PrimitiveTypeUsage", t.toString())) {
                    // System.out.print("\r returnType Resolve Success : " + name);
                    resolveSuccess++;
                } else if (name.equals("String")) {
                    name = "java.lang.String";
                    // System.out.print("\r returnType Resolve Success : " + name);
                    resolveSuccess++;
                } else if (name.equals("void")) {
                    // System.out.print("\r returnType Resolve Success : " + name);
                    resolveSuccess++;
                } else {
                    // System.out.print("\r returnType Resolve Failed : " + name + " @ ");
                    // System.out.print(filePath);
                    // System.out.println(" -> " + t.toString() + "\n");
                    resolveFailed++;
                }
            } catch (Throwable t) {
                // System.out.print("\r returnType Resolve Failed : " + name + " @ ");
                // System.out.print(filePath);
                // System.out.println(" -> " + t.toString() + "\n");
                resolveFailed++;
            }
            arrayList.add(name);
        } else {
            arrayList.add(name);
        }
        returnTypeSet.add(name);
    }

    /**
     * 文字の整形
     * 
     * @param letter
     * @return 整形済みの文字列
     */
    private String formatter(String letter) {
        String[] str = letter.split("");
        StringBuilder value = new StringBuilder();
        for (String tmp : str) {
            if (!tmp.matches("[0-9a-zA-Z._]"))
                break;
            value.append(tmp);
        }
        return value.toString();
    }

}
